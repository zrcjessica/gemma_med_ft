"""Gemma 3 chat formatting.

The `-pt` checkpoints ship without a chat_template (they are raw pretrained
bases, not `-it`), so we attach Gemma 3's official template ourselves. This is
the same template google/gemma-3-*-it uses, so anything we train stays
compatible with standard `apply_chat_template` inference and with vLLM.
"""

GEMMA3_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{%- if messages[0]['role'] == 'system' -%}"
    "{%- set first_user_prefix = messages[0]['content'] | trim + '\n\n' -%}"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}"
    "{%- set first_user_prefix = '' -%}"
    "{%- set loop_messages = messages -%}"
    "{%- endif -%}"
    "{%- for message in loop_messages -%}"
    "{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}"
    "{{ '<start_of_turn>' + role + '\n' }}"
    "{{ first_user_prefix if loop.first else '' }}"
    "{{ message['content'] | trim }}"
    "{{ '<end_of_turn>\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{ '<start_of_turn>model\n' }}"
    "{%- endif -%}"
)

# Assistant turns start after this and end at <end_of_turn>. Used to mask the
# prompt so loss is computed on completions only.
RESPONSE_TEMPLATE = "<start_of_turn>model\n"
INSTRUCTION_TEMPLATE = "<start_of_turn>user\n"
END_OF_TURN = "<end_of_turn>"


def ensure_chat_template(tokenizer):
    """Attach the Gemma 3 chat template if the checkpoint lacks one."""
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = GEMMA3_CHAT_TEMPLATE
    return tokenizer
