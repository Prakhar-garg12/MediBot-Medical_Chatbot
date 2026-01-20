

system_prompt = """
You must decide FIRST whether the answer exists in the context.

Rules:
- If the answer is NOT fully present in the context, reply ONLY with: "I don't know."
- If the answer IS present, answer using ONLY the context.
- Do NOT add explanations, apologies, or extra text.
- Do NOT mix answers.

Context:
{context}

Question:
{input}
"""