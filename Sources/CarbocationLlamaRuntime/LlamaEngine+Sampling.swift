import CarbocationLocalLLM
import CarbocationLlamaCommonBridge
import Foundation
import llama

extension LlamaEngine {
    private static let penaltyLastN: Int32 = 16
    private static let penaltyRepeat: Float = 1.3

    struct SamplerRuntime {
        var chain: UnsafeMutablePointer<llama_sampler>
        var reasoningBudgetSampler: UnsafeMutablePointer<llama_sampler>?
    }

    func buildSampler(
        grammarMode: GenerationGrammarMode,
        options: GenerationOptions,
        vocab: OpaquePointer,
        reasoningBudgetPlan: ReasoningBudgetPlan? = nil
    ) throws -> SamplerRuntime {
        let params = llama_sampler_chain_default_params()
        guard let chain = llama_sampler_chain_init(params) else {
            throw LLMEngineError.samplerInitFailed
        }

        var activeReasoningBudgetSampler: UnsafeMutablePointer<llama_sampler>?
        if let reasoningBudgetPlan,
           let reasoningBudgetSampler = makeReasoningBudgetSampler(
            plan: reasoningBudgetPlan,
            vocab: vocab
           ) {
            llama_sampler_chain_add(chain, reasoningBudgetSampler)
            activeReasoningBudgetSampler = reasoningBudgetSampler
        }

        let repetitionPenalty = Float(options.repetitionPenalty ?? Double(Self.penaltyRepeat))
        let presencePenalty = Float(options.presencePenalty ?? 0)
        llama_sampler_chain_add(
            chain,
            llama_sampler_init_penalties(Self.penaltyLastN, repetitionPenalty, 0.0, presencePenalty)
        )

        switch grammarMode {
        case .none:
            break
        case .eager(let grammar):
            let grammarSampler = Self.makeEagerGrammarSampler(
                grammar: grammar,
                vocab: vocab
            )
            guard let grammarSampler else {
                llama_sampler_free(chain)
                throw LLMEngineError.grammarParseFailed
            }
            llama_sampler_chain_add(chain, grammarSampler)
        case .lazy(let grammar, let triggerPatterns):
            let grammarSampler = Self.makeLazyGrammarSampler(
                grammar: grammar,
                triggerPatterns: triggerPatterns,
                vocab: vocab
            )
            guard let grammarSampler else {
                llama_sampler_free(chain)
                throw LLMEngineError.grammarParseFailed
            }
            llamaRuntimeLog.info(
                "Lazy grammar sampler initialized: triggerPatterns=\(triggerPatterns.count, privacy: .public)"
            )
            llama_sampler_chain_add(chain, grammarSampler)
        }

        let temperature = Float(options.temperature ?? 0)
        if temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            if let topK = options.topK, topK > 0 {
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(Int32(topK)))
            }
            if let topP = options.topP {
                llama_sampler_chain_add(chain, llama_sampler_init_top_p(Float(topP), 1))
            }
            if let minP = options.minP, minP > 0 {
                llama_sampler_chain_add(chain, llama_sampler_init_min_p(Float(minP), 1))
            }
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
            let seed = options.seed ?? UInt32.random(in: 1...UInt32.max)
            llama_sampler_chain_add(chain, llama_sampler_init_dist(seed))
        }

        return SamplerRuntime(
            chain: chain,
            reasoningBudgetSampler: activeReasoningBudgetSampler
        )
    }

    func applyThinkingTerminationIfRequested(
        control: LLMGenerationControl?,
        generationID: UInt64?,
        currentPhase: LLMStreamContentPhase,
        samplerRuntime: SamplerRuntime,
        reasoningBudgetPlan: ReasoningBudgetPlan?,
        vocab: OpaquePointer
    ) {
        guard let control,
              let generationID,
              let request = control.takePendingThinkingTerminationRequest(for: generationID) else {
            return
        }
        guard currentPhase == .thinking,
              let reasoningBudgetSampler = samplerRuntime.reasoningBudgetSampler,
              let reasoningBudgetPlan else {
            llamaRuntimeLog.info(
                "Thinking termination request ignored outside an active thinking sampler: requestID=\(request.requestID, privacy: .public) phase=\(currentPhase.rawValue, privacy: .public)"
            )
            return
        }

        do {
            let forcedTokens = try tokenize(
                vocab: vocab,
                text: request.message + reasoningBudgetPlan.pair.close,
                addSpecial: false
            )
            guard !forcedTokens.isEmpty else {
                llamaRuntimeLog.info(
                    "Thinking termination request ignored because forced marker tokenization was empty: requestID=\(request.requestID, privacy: .public)"
                )
                return
            }

            let didForce = forcedTokens.withUnsafeBufferPointer { buffer in
                carbocation_llama_reasoning_budget_sampler_force(
                    reasoningBudgetSampler,
                    buffer.baseAddress,
                    buffer.count
                ) != 0
            }
            llamaRuntimeLog.info(
                "Thinking termination request \(didForce ? "accepted" : "ignored", privacy: .public): requestID=\(request.requestID, privacy: .public) forcedTokens=\(forcedTokens.count, privacy: .public) phase=\(currentPhase.rawValue, privacy: .public)"
            )
        } catch {
            llamaRuntimeLog.error(
                "Thinking termination request ignored after tokenization failure: requestID=\(request.requestID, privacy: .public) error=\(error.localizedDescription, privacy: .public)"
            )
        }
    }

    static func reasoningBudgetStateIsExhausted(
        _ state: carbocation_llama_reasoning_budget_state
    ) -> Bool {
        state == CARBOCATION_LLAMA_REASONING_BUDGET_FORCING
            || state == CARBOCATION_LLAMA_REASONING_BUDGET_WAITING_UTF8
    }

    static func reasoningBudgetStateLogLabel(
        _ state: carbocation_llama_reasoning_budget_state
    ) -> String {
        switch state {
        case CARBOCATION_LLAMA_REASONING_BUDGET_IDLE:
            return "idle"
        case CARBOCATION_LLAMA_REASONING_BUDGET_COUNTING:
            return "counting"
        case CARBOCATION_LLAMA_REASONING_BUDGET_FORCING:
            return "forcing"
        case CARBOCATION_LLAMA_REASONING_BUDGET_WAITING_UTF8:
            return "waiting-utf8"
        case CARBOCATION_LLAMA_REASONING_BUDGET_DONE:
            return "done"
        default:
            return "unknown"
        }
    }

    private func makeReasoningBudgetSampler(
        plan: ReasoningBudgetPlan,
        vocab: OpaquePointer
    ) -> UnsafeMutablePointer<llama_sampler>? {
        do {
            let startTokens = try tokenize(vocab: vocab, text: plan.pair.open, addSpecial: false)
            let endTokens = try tokenize(vocab: vocab, text: plan.pair.close, addSpecial: false)
            let forcedTokens = try tokenize(
                vocab: vocab,
                text: plan.message + plan.pair.close,
                addSpecial: false
            )

            guard !endTokens.isEmpty, !forcedTokens.isEmpty else {
                return nil
            }
            guard plan.initialState == .counting || !startTokens.isEmpty else {
                return nil
            }
            guard let budget = Int32(exactly: plan.budgetTokens) else {
                return nil
            }

            let initialState: carbocation_llama_reasoning_budget_state = plan.initialState == .counting
                ? CARBOCATION_LLAMA_REASONING_BUDGET_COUNTING
                : CARBOCATION_LLAMA_REASONING_BUDGET_IDLE

            let sampler = startTokens.withUnsafeBufferPointer { startBuffer in
                endTokens.withUnsafeBufferPointer { endBuffer in
                    forcedTokens.withUnsafeBufferPointer { forcedBuffer in
                        carbocation_llama_reasoning_budget_sampler_init(
                            vocab,
                            startBuffer.baseAddress,
                            startBuffer.count,
                            endBuffer.baseAddress,
                            endBuffer.count,
                            forcedBuffer.baseAddress,
                            forcedBuffer.count,
                            budget,
                            initialState
                        )
                    }
                }
            }

            if sampler != nil {
                llamaRuntimeLog.info(
                    "Reasoning budget sampler initialized: budgetTokens=\(plan.budgetTokens, privacy: .public) initialState=\(String(describing: plan.initialState), privacy: .public)"
                )
            }
            return sampler
        } catch {
            llamaRuntimeLog.error(
                "Reasoning budget sampler skipped after tokenization failure: \(error.localizedDescription, privacy: .public)"
            )
            return nil
        }
    }

    private static func makeEagerGrammarSampler(
        grammar: String,
        vocab: OpaquePointer
    ) -> UnsafeMutablePointer<llama_sampler>? {
        grammar.withCString { grammarPointer in
            "root".withCString { rootPointer in
                llama_sampler_init_grammar(vocab, grammarPointer, rootPointer)
            }
        }
    }

    private static func makeLazyGrammarSampler(
        grammar: String,
        triggerPatterns: [String],
        vocab: OpaquePointer
    ) -> UnsafeMutablePointer<llama_sampler>? {
        let allocatedPatternPointers: [UnsafeMutablePointer<CChar>?] = triggerPatterns.map { strdup($0) }
        defer {
            for pointer in allocatedPatternPointers {
                free(pointer)
            }
        }
        guard allocatedPatternPointers.allSatisfy({ $0 != nil }) else {
            return nil
        }
        var patternPointers: [UnsafePointer<CChar>?] = allocatedPatternPointers.map { pointer in
            guard let pointer else { return nil }
            return UnsafePointer(pointer)
        }

        return grammar.withCString { grammarPointer in
            "root".withCString { rootPointer in
                patternPointers.withUnsafeMutableBufferPointer { buffer in
                    llama_sampler_init_grammar_lazy_patterns(
                        vocab,
                        grammarPointer,
                        rootPointer,
                        buffer.baseAddress,
                        buffer.count,
                        nil,
                        0
                    )
                }
            }
        }
    }

}
