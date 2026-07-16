"""Gemma 3 chat formatting.

The `-pt` checkpoints ship without a chat_template (they are raw pretrained
bases, not `-it`), and even the `-it` template needs one addition for training:
TRL's `assistant_only_loss` masks the prompt by calling `apply_chat_template(
return_assistant_tokens_mask=True)`, which only works if the template marks
assistant spans with `{% generation %}`. The stock Gemma 3 template has no such
marker, so we keep two variants:

  GEMMA3_CHAT_TEMPLATE       -- stock. Saved with the model; what vLLM/inference use.
  GEMMA3_CHAT_TEMPLATE_TRAIN -- identical output, plus {% generation %} spans.

Both must render byte-identical text; test_chat.py asserts this.
"""

_PREAMBLE = (
    "{{ bos_token }}"
    "{%- if messages[0]['role'] == 'system' -%}"
    "{%- set first_user_prefix = messages[0]['content'] | trim + '\n\n' -%}"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}"
    "{%- set first_user_prefix = '' -%}"
    "{%- set loop_messages = messages -%}"
    "{%- endif -%}"
)

_TAIL = (
    "{%- if add_generation_prompt -%}"
    "{{ '<start_of_turn>model\n' }}"
    "{%- endif -%}"
)

GEMMA3_CHAT_TEMPLATE = (
    _PREAMBLE
    + "{%- for message in loop_messages -%}"
    "{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}"
    "{{ '<start_of_turn>' + role + '\n' }}"
    "{{ first_user_prefix if loop.first else '' }}"
    "{{ message['content'] | trim }}"
    "{{ '<end_of_turn>\n' }}"
    "{%- endfor -%}" + _TAIL
)

# <end_of_turn> sits inside the generation span on purpose: the model must learn
# to emit it, or it will never stop.
GEMMA3_CHAT_TEMPLATE_TRAIN = (
    _PREAMBLE
    + "{%- for message in loop_messages -%}"
    "{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}"
    "{{ '<start_of_turn>' + role + '\n' }}"
    "{{ first_user_prefix if loop.first else '' }}"
    "{%- if message['role'] == 'assistant' -%}"
    "{% generation %}{{ message['content'] | trim }}{{ '<end_of_turn>\n' }}{% endgeneration %}"
    "{%- else -%}"
    "{{ message['content'] | trim }}"
    "{{ '<end_of_turn>\n' }}"
    "{%- endif -%}"
    "{%- endfor -%}" + _TAIL
)

RESPONSE_TEMPLATE = "<start_of_turn>model\n"
INSTRUCTION_TEMPLATE = "<start_of_turn>user\n"
END_OF_TURN = "<end_of_turn>"


def ensure_chat_template(tokenizer, for_training: bool = False):
    """Attach a Gemma 3 chat template.

    for_training=True installs the {% generation %} variant so TRL can build the
    assistant-token mask. Save the model with for_training=False.
    """
    tokenizer.chat_template = GEMMA3_CHAT_TEMPLATE_TRAIN if for_training else GEMMA3_CHAT_TEMPLATE
    return tokenizer
