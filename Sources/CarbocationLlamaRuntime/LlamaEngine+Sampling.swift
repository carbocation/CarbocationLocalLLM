import CarbocationLocalLLM
import CarbocationLlamaCommonBridge
import Foundation
import llama

package struct LlamaResolvedSamplerDiagnostics: Hashable, Sendable {
    package var requestTemperature: Double?
    package var requestTopK: Int?
    package var requestTopP: Double?
    package var requestMinP: Double?
    package var requestPresencePenalty: Double?
    package var requestRepetitionPenalty: Double?
    package var requestSeed: UInt32?
    package var chain: [String]
    package var resolvedTemperature: Double
    package var resolvedTopK: Int?
    package var resolvedTopP: Double?
    package var resolvedMinP: Double?
    package var penaltyLastN: Int32
    package var repetitionPenalty: Double
    package var frequencyPenalty: Double
    package var presencePenalty: Double
    package var seed: UInt32?

    package var requestLine: String {
        [
            "temperature=\(Self.formatOptional(requestTemperature))",
            "topK=\(Self.formatOptional(requestTopK))",
            "topP=\(Self.formatOptional(requestTopP))",
            "minP=\(Self.formatOptional(requestMinP))",
            "presencePenalty=\(Self.formatOptional(requestPresencePenalty))",
            "repetitionPenalty=\(Self.formatOptional(requestRepetitionPenalty))",
            "seed=\(Self.formatOptional(requestSeed))"
        ].joined(separator: " ")
    }

    package var resolvedLine: String {
        [
            "chain=\(chain.joined(separator: ","))",
            "temperature=\(Self.format(resolvedTemperature))",
            "topK=\(Self.formatOptional(resolvedTopK, nilValue: "disabled"))",
            "topP=\(Self.formatOptional(resolvedTopP, nilValue: "disabled"))",
            "minP=\(Self.formatOptional(resolvedMinP, nilValue: "disabled"))",
            "penaltyLastN=\(penaltyLastN)",
            "repetitionPenalty=\(Self.format(repetitionPenalty))",
            "frequencyPenalty=\(Self.format(frequencyPenalty))",
            "presencePenalty=\(Self.format(presencePenalty))",
            "seed=\(Self.formatOptional(seed, nilValue: "none"))"
        ].joined(separator: " ")
    }

    private static func formatOptional<T>(_ value: T?, nilValue: String = "nil") -> String {
        value.map { "\($0)" } ?? nilValue
    }

    private static func formatOptional(_ value: Double?, nilValue: String = "nil") -> String {
        value.map(format) ?? nilValue
    }

    private static func format(_ value: Double) -> String {
        if value.rounded(.towardZero) == value {
            return String(Int(value))
        }
        return String(value)
    }
}

extension LlamaEngine {
    fileprivate static let penaltyLastN: Int32 = 16
    fileprivate static let penaltyRepeat: Float = 1.3

    struct SamplerRuntime {
        var chain: UnsafeMutablePointer<llama_sampler>
        var reasoningBudgetSampler: UnsafeMutablePointer<llama_sampler>?
    }

    package nonisolated static func resolvedSamplerDiagnostics(
        options: GenerationOptions,
        grammarSampler: String? = nil,
        usesReasoningBudgetSampler: Bool = false
    ) -> LlamaResolvedSamplerDiagnostics {
        let temperature = options.temperature ?? 0
        let repetitionPenalty = options.repetitionPenalty ?? Double(Self.penaltyRepeat)
        let frequencyPenalty = 0.0
        let presencePenalty = options.presencePenalty ?? 0

        var chain: [String] = []
        if usesReasoningBudgetSampler {
            chain.append("reasoning-budget")
        }
        chain.append("penalties")
        if let grammarSampler {
            chain.append("grammar:\(grammarSampler)")
        }

        let resolvedTopK: Int?
        let resolvedTopP: Double?
        let resolvedMinP: Double?
        let resolvedSeed: UInt32?
        if temperature <= 0 {
            chain.append("greedy")
            resolvedTopK = nil
            resolvedTopP = nil
            resolvedMinP = nil
            resolvedSeed = nil
        } else {
            if let topK = options.topK, topK > 0 {
                chain.append("top-k")
                resolvedTopK = topK
            } else {
                resolvedTopK = nil
            }
            if let topP = options.topP {
                chain.append("top-p")
                resolvedTopP = topP
            } else {
                resolvedTopP = nil
            }
            if let minP = options.minP, minP > 0 {
                chain.append("min-p")
                resolvedMinP = minP
            } else {
                resolvedMinP = nil
            }
            chain.append("temperature")
            chain.append("distribution")
            resolvedSeed = options.seed
        }

        return LlamaResolvedSamplerDiagnostics(
            requestTemperature: options.temperature,
            requestTopK: options.topK,
            requestTopP: options.topP,
            requestMinP: options.minP,
            requestPresencePenalty: options.presencePenalty,
            requestRepetitionPenalty: options.repetitionPenalty,
            requestSeed: options.seed,
            chain: chain,
            resolvedTemperature: temperature,
            resolvedTopK: resolvedTopK,
            resolvedTopP: resolvedTopP,
            resolvedMinP: resolvedMinP,
            penaltyLastN: Self.penaltyLastN,
            repetitionPenalty: repetitionPenalty,
            frequencyPenalty: frequencyPenalty,
            presencePenalty: presencePenalty,
            seed: resolvedSeed
        )
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
