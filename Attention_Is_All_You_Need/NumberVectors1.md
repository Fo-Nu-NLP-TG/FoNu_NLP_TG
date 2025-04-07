Why Use a 512-Number Model Vector? Is It by Default?

What is the 512-Number Vector?

In the Transformer code, d_model (like 512) is the size of the vectors representing each word or token.

 For example:
"cat" → [0.1, -0.5, 0.3, ..., 0.2] (512 numbers).

These vectors are the hidden states—numerical summaries of a word’s meaning, context, or role in a sentence—produced by layers like src_embed or tgt_embed.

Why 512?

Not Default, but Common: The number 512 isn’t a universal default—it’s a design choice from the original "Attention is All You Need" Transformer paper (2017). 

The authors picked it as a balance between:

- Expressiveness: More numbers (higher d_model) let the vector capture more details about a word—like its meaning, grammar, or relationship to other words.

- Computation Cost: Bigger vectors mean more calculations (e.g., matrix multiplications). 512 is large enough to work well but not so huge that it slows training too much.

- Historical Context: 
Early neural networks (e.g., RNNs) used smaller sizes (like 128 or 256).

Transformers, being more parallel and powerful, bumped it to 512 for better performance on tasks like translation.

Is It Always 512?

No: 

It’s configurable. 

In practice:

- Small models (e.g., BERT-small) use 256 or 128.

- Large models (e.g., GPT-3) use 768, 1024, or even 12,288!

