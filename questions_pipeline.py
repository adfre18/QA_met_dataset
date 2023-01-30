from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelWithLMHead
import random
from tqdm import tqdm


class QuestionGenerator(object):
    def __init__(self, config):
        self.config = config
        self.input_filename = config["DatasetsSettings"]["artists_info_dataset_filename"]
        self.output_filename = config["DatasetsSettings"]["questions_filename"]
        self.__load_pretrained_models()

    def generate_questions(self):
        # using HF pipeline for generating NE - then creating questions
        contexts = self.__load_contexts()
        with open(self.output_filename, 'w+', encoding='utf-8') as fw:
            for artist_name, context in tqdm(contexts.items(), desc="Generating questions"):
                nlp = pipeline('ner', model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
                output = nlp(context)
                random_ne = random.randint(0, len(output)-1)
                answer = output[random_ne]['word']
                question = self.__get_question(answer, context)
                print(artist_name, question, answer, context.replace("\n", ""), file=fw, sep="\t")
                fw.flush()
    
    def __load_pretrained_models(self):
        # Loading model for NER task
        self.model = AutoModelForTokenClassification.from_pretrained(self.config["Models"]["ner"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["Models"]["ner"])

        # Loading model for Question generating task
        self.tokenizer_question = AutoTokenizer.from_pretrained(self.config["Models"]["question_generator"])
        self.model_question = AutoModelWithLMHead.from_pretrained(self.config["Models"]["question_generator"])
    
    def __load_contexts(self):
        # load contexts for generating questions
        contexts = dict()
        with open(self.input_filename, 'r+', encoding='utf-8') as fr:
            lines = fr.readlines()
        for line in lines:
            contexts[line.split("\t")[0]] = line.split("\t")[1]
        return contexts

    def __get_question(self, answer, context, max_length=64):
        # generating questions according to the context and desired answer
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.tokenizer_question([input_text], return_tensors='pt')
        output = self.model_question.generate(input_ids=features['input_ids'],
                                              attention_mask=features['attention_mask'],
                                              max_length=max_length)
        question = self.tokenizer_question.decode(output[0], skip_special_tokens=True)
        question = question.replace("<pad> question: ", "").replace("</s>", "")
        return question


class QuestionsAnswering(object):
    def __init__(self, config):
        self.config = config
        self.input_filename = config["DatasetsSettings"]["questions_filename"]
        self.output_filename = config["DatasetsSettings"]["predicted_answers_filename"]
        self.__load_data()
        self.__load_pretrained_model()

    def __load_pretrained_model(self):   
        # loading pretrained model for question answering
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["Models"]["qa"])
        self.model = AutoModelWithLMHead.from_pretrained(self.config["Models"]["qa"])

    def __load_data(self):
        # loading questions from file
        with open(self.input_filename, "r+", encoding='utf-8') as fr:
            self.data_questions = fr.readlines()

    def generate_answers(self):
        # generating answers
        with open(self.output_filename, 'w+', encoding='utf-8') as fw:
            for line_questions in tqdm(self.data_questions, desc="Generating answers"):
                artist_name = line_questions.split("\t")[0]

                question = line_questions.split("\t")[1]
                reference_answer = line_questions.split("\t")[2]
                context = line_questions.split("\t")[3]
                # create model input in specific format
                model_input = f"question: {question} context: {context}"
                encoded_input = self.tokenizer([model_input], return_tensors='pt', max_length=512, truncation=True)
                output = self.model.generate(input_ids=encoded_input.input_ids,
                                             attention_mask=encoded_input.attention_mask)
                # decode model output
                output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                idx = 0
                if output.rstrip() != reference_answer.rstrip():
                    idx += 1
                context = context.replace("\n", "")
                # writing to file
                print(f"{artist_name}", f"Question: {question}", f"Ref: {reference_answer.rstrip()}",
                      f"Pred: {output.rstrip()}", f"Context: {context}", file=fw, sep="\t")
                fw.flush()
