from dataclasses import dataclass, field
from typing import List

QUESTION_INSTRUCTION = (
    "The user is looking at a remote sensing image. "
    "The assistant asks him simple questions to explore the contents of the image, and the user answers. "
    "It must be possible to answer to the questions only by looking at the image. "
    "The assistant asks one question at a time, without giving the answer. "
    "The assistant avoids binary questions when possible."
)

ANSWER_INSTRUCTION = (
    "Answer given questions. "
    "If you are not sure about the answer, say you don't know honestly. "
    "Don't imagine any contents that are not in the image."
)

SUMMARY_INSTRUCTION = (
    "Summarize the information in the chat between the assistant and the user, creating a descriptive paragraph of the contents of the image. "
    "Don't add information. Don't miss information."
)

CHANGE_INSTRUCTION = (
    "I will provide descriptions of two remote sensing images taken at the same location at different times. "
    "Extracting information from the descriptions, generate a textual paragraph summarizing the changes occurred, if any. "
    "Summarize the changes without focusing on the single description, but directly extracting only the changes "
)


@dataclass
class Conversation:
    """A class that keeps all conversation history."""

    q_system: str = QUESTION_INSTRUCTION
    a_system: str = ANSWER_INSTRUCTION
    s_system: str = SUMMARY_INSTRUCTION
    c_c_system: str = CHANGE_INSTRUCTION
    roles: tuple[str] = ("USER", "ASSISTANT")
    messages: List[List[str]] = field(default_factory=list)
    sep: str = " "
    sep2: str = "</s>"

    def reset(self):
        self.messages = []

    def manipulate_prompt_API(self, prompt):
        new_prompt = prompt[0] + ". Provide next question."
        return new_prompt

    def get_vicuna_question_prompt_API(self):
        history = {"internal": [], "visible": []}
        # For now I will append on both internal and visible
        previous_role = "USER"  # The first question is always from the assistant!
        chat_round = [self.q_system]
        if len(self.messages) != 0:
            for _, (role, message) in enumerate(self.messages):
                if role != previous_role:
                    if message == "":
                        print("Warning, empty message found!")
                    chat_round.append(message)
                    previous_role = role
                    # Append the chat round
                    if len(chat_round) == 2:
                        history["internal"].append(chat_round)
                        history["visible"].append(chat_round)
                        chat_round = []
                else:
                    raise ValueError(
                        "The messages should alternate between the two roles!"
                    )
            return history, chat_round
        else:
            raise ValueError(
                "In this beta version history cannot be empty! Please provide the template initial question and answer pair."
            )

    def get_vicuna_question_prompt(self):
        if len(self.messages) != 0:
            seps = [self.sep, self.sep2]
            prompt = self.q_system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message != "":
                    prompt += role + ": " + message + seps[i % 2]
                else:
                    print("Warning, empty message found!")
            # Append the last role
            prompt += (
                "Provide next question. Avoid asking yes/no questions. "
                + self.roles[1]
                + ": Question:"
            )
            return prompt
        else:
            raise ValueError(
                "In this beta version history cannot be empty! Please provide the template initial question and answer pair."
            )

    def get_vicuna_summary_prompt(self):  # FIXME
        if len(self.messages) != 0:
            seps = [self.sep, self.sep2]
            prompt = self.q_system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message != "":
                    prompt += role + ": " + message + seps[i % 2]
                else:
                    print("Warning, empty message found!")

            prompt += self.s_system + seps[0] + self.roles[1] + ":"

            return prompt
        else:
            raise ValueError(
                "In this beta version history cannot be empty! Please provide the template initial question and answer pair."
            )

    def get_vicuna_change_captioning_prompt(self, description1, description2):
        if description1 != "" and description2 != "":
            seps = [self.sep, self.sep2]
            prompt = self.c_c_system + seps[0]
            prompt += self.roles[0] + ": Description 1: " + description1 + seps[0]
            prompt += " Description 2: " + description2 + seps[0]

            prompt += self.roles[1] + ": Changes:"

            return prompt
        else:
            raise ValueError("At least one summary is empty!")

    def get_vicuna_change_captioning_prompt_API(self, description1, description2):
        history = {"internal": [], "visible": []}
        if description1 != "" and description2 != "":
            prompt = self.c_c_system
            prompt += " Description 1: " + description1
            prompt += " Description 2: " + description2

            return history, prompt
        else:
            raise ValueError("At least one summary is empty!")

    def get_blip2_prompt(self, blip_context=1):
        prompt = self.a_system + self.sep
        if len(self.messages) != 0 and len(self.messages) >= blip_context:
            for role, message in self.messages[-blip_context:]:
                if message != "":
                    if role == self.roles[1]:
                        prompt += message + self.sep
                    if role == self.roles[0]:
                        prompt += message + self.sep
                else:
                    raise ValueError("History cannot contain empty messages!")
            # Append the last role
            prompt += "Answer:"
            return prompt
        else:
            raise ValueError("Question not found")

    def append_message(self, role: str, message: str):
        self.messages.append([role, message])

    def append_question(self, question: str):
        message = question
        if len(self.messages) == 0 or self.messages[-1][0] == self.roles[0]:
            # Append if the list is empty or the last message is from the user (so an answer)
            self.append_message(role=self.roles[1], message=message)
        else:
            raise ValueError("The last message must be an answer from the user!")

    def append_answer(self, answer: str):
        message = answer
        if len(self.messages) != 0 and self.messages[-1][0] == self.roles[1]:
            # Append if the list is not empty and the last message is from the assistant (so a question)
            self.append_message(role=self.roles[0], message=message)
        else:
            raise ValueError("The last message must be a question from the assistant!")

    def return_messages(self):
        """
        Returns two lists, one for the questions and the other for the answers
        """
        questions = []
        answers = []
        for message in self.messages:
            if message[0] == self.roles[1]:
                questions.append(message[1])
            elif message[0] == self.roles[0]:
                answers.append(message[1])
            else:
                raise ValueError("Role not recognized!")

        return questions, answers


if __name__ == "__main__":
    conv_v1 = Conversation(
        q_system=(
            "I have an image. "
            "Ask me questions about the content of this image. "
            "Carefully asking me informative questions to maximize your information about this image content. "
            "Each time ask one question only without giving an answer. "
            "Avoid asking yes/no questions."
            'I\'ll put my answer beginning with "Answer:".'
        ),
        a_system="Answer given questions. If you are not sure about the answer, say you don't know honestly. Don't imagine any contents that are not in the image.",
        s_system=SUMMARY_INSTRUCTION,
        roles=("USER", "ASSISTANT"),
        messages=[],
        sep=" ",
    )
    question = "Describe this image in detail."
    conv_v1.append_question(question)
    conv_v1.append_answer("An image blue")

    conv_v1.append_question("What is the color of the sky?")
    conv_v1.append_answer("Blue")

    conv_v1.append_question("What is the color of the sky?")
    conv_v1.append_answer("Blue")

    print(conv_v1.get_vicuna_question_prompt())
