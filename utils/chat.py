from utils.blip2 import Blip2
from utils.vicuna import load_model
from utils.vicuna import generate_stream
from utils.conversation import Conversation
from text_generator_api import prompt_response_chat

FIRST_QUESTION = "Describe this remote sensed image in detail."

QUESTIONER_MODELS = ["chatgpt", "vicuna"]
ANSWERER_MODELS = ["blip2"]
SUMMARIZERS = ["chatgpt", "vicuna"]


def load_vicuna(device: str):
    model_name = "TheBloke_vicuna-13B-1.1-GPTQ-4bit-128g"
    model_path = (
        "D:/Riccardo/Second year/open_source_llms/FastChat/models/" + model_name
    )
    tokenizer, model, context_len = load_model(model_path, device)

    return model, tokenizer, context_len


def load_blip2(device: str):
    blip2 = Blip2("FlanT5 XXL", device=device, bit8=True)

    return blip2


def load_questioner(model_name: str, device: str):
    assert device is not None
    if model_name in QUESTIONER_MODELS:
        if model_name == "chatgpt":
            raise NotImplementedError("chatgpt is not implemented yet")
        elif model_name == "vicuna":
            model, tokenizer, context_len = load_vicuna(device)
        else:
            raise ValueError("{} is not a valid question model".format(model_name))

    return model, tokenizer


def load_answerer(model_name: str, device: str):
    assert device is not None
    if model_name in ANSWERER_MODELS:
        if model_name == "blip2":
            model = load_blip2(device)
        else:
            raise ValueError("{} is not a valid question model".format(model_name))

    return model


class Chatter:
    """
    This class serves as the chatbot to ask questions about an image using questioner model and answer them with answerer model
    """
    
    def __init__(
        self,
        params = None
        # questioner=None,
        # answerer=None,
        # summarizer=None,
        # q_device=None,
        # a_device=None,
        # s_device=None,
        # q_maxtok=30,
        # a_maxtok=30,
        # s_maxtok=100,
        # q_context=2048,
        # a_context=1,
        # s_context=2048,
    ):
        if params is None:
            raise ValueError("No parameters given")
        
        if params["questioner_type"] is not None:
            # Load the questioner
            print("Loading questioner {}", params["questioner_type"])
            self.questioner, self.q_tokenizer = load_questioner(params["questioner_type"], params["questioner_device"])
        if answerer is not None:
            print("Loading answerer..")
            self.answerer = load_answerer(answerer, a_device)

        if summarizer is not None:
            if summarizer != questioner:
                print("Loading summarizer..")
                self.summarizer, self.s_tokenizer = load_questioner(
                    summarizer, s_device
                )  # FIXME rivedere
            else:
                self.summarizer = self.questioner
                self.s_tokenizer = self.q_tokenizer

        self.q_device = q_device
        self.a_device = a_device
        self.s_device = s_device
        self.q_maxtok = q_maxtok
        self.a_maxtok = a_maxtok
        self.s_maxtok = s_maxtok
        self.q_context = q_context
        self.a_context = a_context
        self.s_context = s_context
        self.cc_maxtok = 200  # CHANGE

        self.conversation = Conversation(
            roles=("USER", "ASSISTANT"),
            messages=[],
            sep=" ",
        )

    def reset_history(self):
        self.conversation.reset()

    def call_questioner(self, prompt):
        params = {
            "prompt": prompt,
            "max_new_tokens": self.q_maxtok,
            "temperature": 1.0,
            "stop": "</s>",
        }
        question = generate_stream(
            params, self.questioner, self.q_tokenizer, self.q_context, self.q_device
        )
        return question

    def call_summarizer(self, prompt):
        params = {
            "prompt": prompt,
            "max_new_tokens": self.s_maxtok,
            "temperature": 1.0,
            "stop": "</s>",
        }
        summary = generate_stream(
            params, self.summarizer, self.s_tokenizer, self.s_context, self.s_device
        )
        return summary

    def call_change_captioner(self, prompt):
        params = {
            "prompt": prompt,
            "max_new_tokens": self.cc_maxtok,
            "temperature": 1.0,
            "stop": "</s>",
        }
        change_summary = generate_stream(
            params, self.summarizer, self.s_tokenizer, self.s_context, self.s_device
        )
        return change_summary


    def ask_question_API(self) -> str:
        """
        This function asks a question about the image using the questioner model and the previous chat context
        """
        if len(self.conversation.messages) == 0:
            # first question is given by human to request a general discription
            question = FIRST_QUESTION
        else:
            print("Asking question..")
            history, prompt = self.conversation.get_vicuna_question_prompt_API()
            prompt = self.conversation.manipulate_prompt_API(prompt)
            question = prompt_response_chat(prompt, history)["visible"][-1][1]

        return question

    def question_trim(self, question):
        question = question.replace("\n", " ").strip()
        if "user:" in question:  # Some models make up an answer after asking. remove it
            q = question.split("user:")[0]
            if (
                len(q) == 0
            ):  # some not so clever models will put the question after 'Answer:'.
                raise ValueError("Question not found")
            else:
                question = q.strip()
        return question

    def answer_question(self, image):
        """
        This function answers the question using the answerer model and the previous chat context
        """
        print("Answering question..")
        prompt = self.conversation.get_blip2_prompt(blip_context=self.a_context)
        answer = self.answerer.ask(image, prompt)

        return answer

    def answer_trim(self, answer):
        '''
        It helps in cases in which the answerer model generates a follow-up question after answering given question
        '''
        answer = answer.split("Question:")[0].replace("\n", " ").strip()
        return answer

    def summarize(self):
        if len(self.conversation.messages) == 0:
            raise ValueError("No messages in the conversation")
        else:
            summarizer_prompt = self.conversation.get_vicuna_summary_prompt()
        # summary = self.call_summarizer(summarizer_prompt)
        return summarizer_prompt

    def change_description(self, summary1, summary2):
        history, prompt = self.conversation.get_vicuna_change_captioning_prompt_API(
            summary1, summary2
        )

        change_summary = prompt_response_chat(prompt, history)["visible"][-1][1]
        return change_summary
