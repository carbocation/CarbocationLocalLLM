#include "CarbocationLlamaCommonBridge.h"

#include "chat.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

using bridge_json = nlohmann::ordered_json;

struct carbocation_llama_chat_templates {
    common_chat_templates_ptr value;
};

struct carbocation_llama_chat_plan {
    common_chat_params value;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    bool is_continuation = false;
    bool grammar_needs_prefill = false;
};

struct carbocation_llama_chat_parser {
    common_chat_parser_params params;
    common_chat_msg previous;
    std::string generated_text;
};

namespace {

std::string string_or_empty(const char * value) {
    return value == nullptr ? std::string() : std::string(value);
}

bool copy_string(const std::string & value, char ** output) {
    if (output == nullptr) {
        return false;
    }
    *output = static_cast<char *>(std::malloc(value.size() + 1));
    if (*output == nullptr) {
        return false;
    }
    std::memcpy(*output, value.data(), value.size());
    (*output)[value.size()] = '\0';
    return true;
}

carbocation_llama_chat_status fail(
    carbocation_llama_chat_status status,
    const std::string & message,
    char ** out_error
) {
    if (out_error != nullptr) {
        *out_error = nullptr;
        if (!copy_string(message, out_error)) {
            return CARBOCATION_LLAMA_CHAT_STATUS_ALLOCATION_ERROR;
        }
    }
    return status;
}

void clear_output(char ** output) {
    if (output != nullptr) {
        *output = nullptr;
    }
}

bridge_json tool_call_json(const common_chat_tool_call & call) {
    bridge_json result = {
        {"name", call.name},
        {"arguments", call.arguments},
    };
    if (!call.id.empty()) {
        result["id"] = call.id;
    }
    return result;
}

bridge_json message_json(const common_chat_msg & message) {
    bridge_json calls = bridge_json::array();
    for (const auto & call : message.tool_calls) {
        calls.push_back(tool_call_json(call));
    }

    bridge_json result = {
        {"role", message.role.empty() ? "assistant" : message.role},
        {"content", message.content},
        {"reasoning_content", message.reasoning_content},
        {"tool_calls", std::move(calls)},
    };
    if (!message.tool_name.empty()) {
        result["tool_name"] = message.tool_name;
    }
    if (!message.tool_call_id.empty()) {
        result["tool_call_id"] = message.tool_call_id;
    }
    return result;
}

bridge_json diff_json(const common_chat_msg_diff & diff) {
    bridge_json result = {
        {"reasoning_content_delta", diff.reasoning_content_delta},
        {"content_delta", diff.content_delta},
    };
    if (diff.tool_call_index != std::string::npos) {
        result["tool_call_index"] = diff.tool_call_index;
        result["tool_call_delta"] = tool_call_json(diff.tool_call_delta);
    }
    return result;
}

bool has_only_valid_tool_calls(const common_chat_msg & message) {
    for (const auto & call : message.tool_calls) {
        if (call.name.empty()) {
            return false;
        }
        try {
            const auto arguments = bridge_json::parse(call.arguments);
            if (!arguments.is_object()) {
                return false;
            }
        } catch (const bridge_json::exception &) {
            return false;
        }
    }
    return true;
}

const char * trigger_type_name(common_grammar_trigger_type type) {
    switch (type) {
        case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN: return "token";
        case COMMON_GRAMMAR_TRIGGER_TYPE_WORD: return "word";
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN: return "pattern";
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: return "pattern_full";
    }
    return "unknown";
}

bridge_json render_result_json(
    const carbocation_llama_chat_templates & templates,
    const common_chat_params & params,
    bool grammar_needs_prefill
) {
    bridge_json triggers = bridge_json::array();
    for (const auto & trigger : params.grammar_triggers) {
        bridge_json item = {
            {"type", trigger_type_name(trigger.type)},
            {"value", trigger.value},
        };
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            item["token"] = trigger.token;
        }
        triggers.push_back(std::move(item));
    }

    return {
        {"prompt", params.prompt},
        {"format", common_chat_format_name(params.format)},
        {"format_code", static_cast<int>(params.format)},
        {"grammar", params.grammar},
        {"grammar_lazy", params.grammar_lazy},
        {"grammar_needs_prefill", grammar_needs_prefill},
        {"generation_prompt", params.generation_prompt},
        {"supports_thinking", params.supports_thinking},
        {"thinking_start_tag", params.thinking_start_tag},
        {"thinking_end_tags", params.thinking_end_tags},
        {"grammar_triggers", std::move(triggers)},
        {"preserved_tokens", params.preserved_tokens},
        {"additional_stops", params.additional_stops},
        {"has_parser", !params.parser.empty()},
        {"has_explicit_template", common_chat_templates_was_explicit(templates.value.get())},
    };
}

common_chat_tool_choice parse_tool_choice(
    bridge_json & tools,
    const bridge_json & choice
) {
    if (choice.is_null()) {
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
    if (choice.is_string()) {
        const std::string value = choice;
        if (value == "auto") return COMMON_CHAT_TOOL_CHOICE_AUTO;
        if (value == "none") return COMMON_CHAT_TOOL_CHOICE_NONE;
        if (value == "required") return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        throw std::invalid_argument("Unknown tool choice: " + value);
    }
    if (!choice.is_object() || choice.value("type", "") != "named") {
        throw std::invalid_argument("Tool choice must be auto, none, required, or a named choice");
    }

    const std::string name = choice.value("name", "");
    if (name.empty()) {
        throw std::invalid_argument("Named tool choice is missing a name");
    }

    bridge_json filtered = bridge_json::array();
    for (const auto & tool : tools) {
        if (tool.value("type", "") == "function" &&
            tool.contains("function") &&
            tool.at("function").value("name", "") == name) {
            filtered.push_back(tool);
        }
    }
    if (filtered.empty()) {
        throw std::invalid_argument("Named tool choice does not match an available tool: " + name);
    }
    tools = std::move(filtered);
    return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
}

common_chat_templates_inputs parse_render_inputs(const bridge_json & request) {
    if (!request.is_object()) {
        throw std::invalid_argument("Chat render request must be a JSON object");
    }

    common_chat_templates_inputs inputs;
    const bridge_json messages = request.value("messages", bridge_json::array());
    bridge_json tools = request.value("tools", bridge_json::array());
    inputs.messages = common_chat_msgs_parse_oaicompat(messages);
    inputs.tool_choice = parse_tool_choice(tools, request.value("tool_choice", bridge_json("auto")));
    inputs.tools = common_chat_tools_parse_oaicompat(tools);
    inputs.parallel_tool_calls = request.value("parallel_tool_calls", !inputs.tools.empty());
    inputs.add_generation_prompt = request.value("add_generation_prompt", true);
    inputs.use_jinja = request.value("use_jinja", true);
    inputs.enable_thinking = request.value("enable_thinking", true);
    inputs.force_pure_content = request.value("force_pure_content", false);
    inputs.grammar = request.value("grammar", "");
    inputs.json_schema = request.value("json_schema", "");
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    if (request.contains("continue_final_message")) {
        inputs.continue_final_message = common_chat_continuation_parse(request.at("continue_final_message"));
    }
    if (request.contains("reasoning_effort") && !request.at("reasoning_effort").is_null()) {
        const std::string effort = request.at("reasoning_effort").get<std::string>();
        if (!effort.empty() && effort != "none") {
            inputs.chat_template_kwargs["reasoning_effort"] = bridge_json(effort).dump();
        } else if (effort == "none") {
            inputs.enable_thinking = false;
        }
    }
    if (request.contains("preserve_thinking")) {
        inputs.chat_template_kwargs["preserve_reasoning"] = request.at("preserve_thinking").dump();
    }
    if (request.contains("template_kwargs")) {
        const auto & kwargs = request.at("template_kwargs");
        if (!kwargs.is_object()) {
            throw std::invalid_argument("template_kwargs must be a JSON object");
        }
        for (const auto & item : kwargs.items()) {
            inputs.chat_template_kwargs[item.key()] = item.value().dump();
        }
    }
    if (request.contains("unix_time")) {
        const auto seconds = request.at("unix_time").get<int64_t>();
        inputs.now = std::chrono::system_clock::time_point(std::chrono::seconds(seconds));
    }
    return inputs;
}

} // namespace

extern "C" void carbocation_llama_chat_string_free(char * string) {
    std::free(string);
}

extern "C" carbocation_llama_chat_status carbocation_llama_chat_templates_create(
    const llama_model * model,
    const char * template_override,
    const char * bos_token_override,
    const char * eos_token_override,
    carbocation_llama_chat_templates ** out_templates,
    char ** out_error
) {
    clear_output(out_error);
    if (out_templates == nullptr) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_INVALID_ARGUMENT, "out_templates is required", out_error);
    }
    *out_templates = nullptr;
    if (model == nullptr && (template_override == nullptr || template_override[0] == '\0')) {
        return fail(
            CARBOCATION_LLAMA_CHAT_STATUS_INVALID_ARGUMENT,
            "A model or template override is required",
            out_error
        );
    }

    try {
        auto result = std::make_unique<carbocation_llama_chat_templates>();
        result->value = common_chat_templates_init(
            model,
            string_or_empty(template_override),
            string_or_empty(bos_token_override),
            string_or_empty(eos_token_override)
        );
        if (!result->value) {
            return fail(CARBOCATION_LLAMA_CHAT_STATUS_TEMPLATE_ERROR, "Unable to initialize chat templates", out_error);
        }
        *out_templates = result.release();
        return CARBOCATION_LLAMA_CHAT_STATUS_OK;
    } catch (const std::invalid_argument & error) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_INVALID_ARGUMENT, error.what(), out_error);
    } catch (const std::bad_alloc &) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_ALLOCATION_ERROR, "Unable to allocate chat templates", out_error);
    } catch (const std::exception & error) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_TEMPLATE_ERROR, error.what(), out_error);
    } catch (...) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_INTERNAL_ERROR, "Unknown chat template error", out_error);
    }
}

extern "C" void carbocation_llama_chat_templates_free(carbocation_llama_chat_templates * templates) {
    delete templates;
}

extern "C" bool carbocation_llama_chat_templates_has_explicit_template(
    const carbocation_llama_chat_templates * templates
) {
    return templates != nullptr && templates->value &&
        common_chat_templates_was_explicit(templates->value.get());
}

extern "C" carbocation_llama_chat_status carbocation_llama_chat_templates_render(
    const carbocation_llama_chat_templates * templates,
    const char * request_json,
    carbocation_llama_chat_plan ** out_plan,
    char ** out_result_json,
    char ** out_error
) {
    clear_output(out_result_json);
    clear_output(out_error);
    if (templates == nullptr || !templates->value || request_json == nullptr ||
        out_plan == nullptr || out_result_json == nullptr) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_INVALID_ARGUMENT, "Invalid chat render arguments", out_error);
    }
    *out_plan = nullptr;

    try {
        const bridge_json request = bridge_json::parse(request_json);
        common_chat_templates_inputs inputs = parse_render_inputs(request);
        auto result = std::make_unique<carbocation_llama_chat_plan>();
        result->is_continuation = inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE;
        result->value = common_chat_templates_apply(templates->value.get(), inputs);
        // llama.cpp only prefills template-generated output/tool grammars. A grammar
        // supplied by the application is already positioned at its root.
        result->grammar_needs_prefill = inputs.grammar.empty() && !result->value.grammar.empty();
        if (result->value.prompt.empty()) {
            return fail(CARBOCATION_LLAMA_CHAT_STATUS_TEMPLATE_ERROR, "Chat template rendered an empty prompt", out_error);
        }

        const std::string output = render_result_json(
            *templates,
            result->value,
            result->grammar_needs_prefill
        ).dump();
        if (!copy_string(output, out_result_json)) {
            return fail(CARBOCATION_LLAMA_CHAT_STATUS_ALLOCATION_ERROR, "Unable to allocate render result", out_error);
        }
        *out_plan = result.release();
        return CARBOCATION_LLAMA_CHAT_STATUS_OK;
    } catch (const bridge_json::exception & error) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_INVALID_ARGUMENT, error.what(), out_error);
    } catch (const std::invalid_argument & error) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_INVALID_ARGUMENT, error.what(), out_error);
    } catch (const std::bad_alloc &) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_ALLOCATION_ERROR, "Unable to allocate chat render plan", out_error);
    } catch (const std::exception & error) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_TEMPLATE_ERROR, error.what(), out_error);
    } catch (...) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_INTERNAL_ERROR, "Unknown chat render error", out_error);
    }
}

extern "C" void carbocation_llama_chat_plan_free(carbocation_llama_chat_plan * plan) {
    delete plan;
}

extern "C" carbocation_llama_chat_status carbocation_llama_chat_parser_create(
    const carbocation_llama_chat_plan * plan,
    carbocation_llama_chat_parser ** out_parser,
    char ** out_error
) {
    clear_output(out_error);
    if (plan == nullptr || out_parser == nullptr) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_INVALID_ARGUMENT, "Invalid chat parser arguments", out_error);
    }
    *out_parser = nullptr;

    try {
        auto parser = std::make_unique<carbocation_llama_chat_parser>();
        parser->params.format = plan->value.format;
        parser->params.reasoning_format = plan->reasoning_format;
        parser->params.generation_prompt = plan->value.generation_prompt;
        parser->params.parse_tool_calls = true;
        parser->params.is_continuation = plan->is_continuation;
        if (!plan->value.parser.empty()) {
            parser->params.parser.load(plan->value.parser);
        }
        if (parser->params.is_continuation) {
            parser->previous = common_chat_parse("", true, parser->params);
        }
        *out_parser = parser.release();
        return CARBOCATION_LLAMA_CHAT_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_ALLOCATION_ERROR, "Unable to allocate chat parser", out_error);
    } catch (const std::exception & error) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_PARSE_ERROR, error.what(), out_error);
    } catch (...) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_INTERNAL_ERROR, "Unknown chat parser error", out_error);
    }
}

extern "C" void carbocation_llama_chat_parser_free(carbocation_llama_chat_parser * parser) {
    delete parser;
}

extern "C" carbocation_llama_chat_status carbocation_llama_chat_parser_update(
    carbocation_llama_chat_parser * parser,
    const char * utf8_chunk,
    bool is_partial,
    char ** out_result_json,
    char ** out_error
) {
    clear_output(out_result_json);
    clear_output(out_error);
    if (parser == nullptr || utf8_chunk == nullptr || out_result_json == nullptr) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_INVALID_ARGUMENT, "Invalid chat parser update arguments", out_error);
    }

    try {
        const std::string generated_text = parser->generated_text + utf8_chunk;
        common_chat_msg parsed = common_chat_parse(generated_text, is_partial, parser->params);
        bool used_raw_content_fallback = false;
        if (!is_partial && (parsed.empty() || !has_only_valid_tool_calls(parsed))) {
            parsed = {};
            parsed.role = "assistant";
            parsed.content = generated_text;
            used_raw_content_fallback = true;
        }

        std::vector<common_chat_msg_diff> diffs;
        if (!parsed.empty()) {
            diffs = common_chat_msg_diff::compute_diffs(parser->previous, parsed);
        }

        bridge_json diffs_json = bridge_json::array();
        for (const auto & diff : diffs) {
            diffs_json.push_back(diff_json(diff));
        }
        bridge_json result = {
            {"message", message_json(parsed.empty() ? parser->previous : parsed)},
            {"diffs", std::move(diffs_json)},
            {"is_partial", is_partial},
            {"used_raw_content_fallback", used_raw_content_fallback},
        };
        if (!copy_string(result.dump(), out_result_json)) {
            return fail(CARBOCATION_LLAMA_CHAT_STATUS_ALLOCATION_ERROR, "Unable to allocate parser result", out_error);
        }
        parser->generated_text = generated_text;
        if (!parsed.empty()) {
            parser->previous = std::move(parsed);
        }
        return CARBOCATION_LLAMA_CHAT_STATUS_OK;
    } catch (const bridge_json::exception & error) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_PARSE_ERROR, error.what(), out_error);
    } catch (const std::bad_alloc &) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_ALLOCATION_ERROR, "Unable to allocate parser state", out_error);
    } catch (const std::exception & error) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_PARSE_ERROR, error.what(), out_error);
    } catch (...) {
        return fail(CARBOCATION_LLAMA_CHAT_STATUS_INTERNAL_ERROR, "Unknown chat parser error", out_error);
    }
}
