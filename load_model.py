from transformers import AutoModelForCausalLM, GPTQConfig, AutoTokenizer, GenerationConfig
import time
if __name__=="__main__":
    model_name_or_path = "lmsys/vicuna-13b-v1.5" # TheBloke/vicuna-13B-v1.5-GPTQ
    # To use a different branch, change revision
    # For example: revision="main"
    gen_cfg = GenerationConfig.from_pretrained(model_name_or_path)
    gen_cfg.max_new_tokens=200
    gen_cfg.do_sample=True
    gen_cfg.temperature=0.3
    gen_cfg.top_p=0.95
    gen_cfg.top_k=40
    gen_cfg.repetition_penalty=1.1
    # Quantization config 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    #gptq_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer, disable_exllama=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="cuda:0",
                                                trust_remote_code=False,
                                                revision="main")
    model.generation_config = gen_cfg
    #config = AutoConfig.from_pretrained(model_name_or_path)
    # print(config)
    #gen_cfg, unused = GenerationConfig.from_pretrained(model_name_or_path, max_new_tokens=200, do_sample=True, temperature=0.7,top_p=0.95,top_k=40, repetition_penalty=1.1, return_unused_kwargs=True)
    # print(gen_cfg)
    # # # print(gen_cfg)
    # model.generation_config = gen_cfg
    # print(model.generation_config)

    prompt = "Tell me about AI."
    prompt_template=[f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:''']
    
    # Inference can also be done using transformers' pipeline
    print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, padding=True, return_tensors='pt').input_ids.to(model.device)
    start = time.time()
    output = model.generate(inputs=input_ids)
    end = time.time()
    print(f"Time taken: {end-start}")
    print(tokenizer.batch_decode(output, skip_special_tokens=True))

    # print("*** Pipeline:")
    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     generation_config=gen_cfg
    # )

    # print(pipe(prompt_template)[0]['generated_text'])
    

# A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

# USER: Hello!
# ASSISTANT: Hello!</s>
# USER: How are you?
# ASSISTANT: I am good.</s>