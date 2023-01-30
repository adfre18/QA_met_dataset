from wiki_scraping import WikiDataLoader
from questions_pipeline import QuestionGenerator, QuestionsAnswering
import configparser
import os
BASEPATH = os.path.dirname(__file__)

def load_config_file():
    config_file = configparser.ConfigParser()

    # READ CONFIG FILE
    config_file.read(os.path.join(BASEPATH, "configurations.ini"))
    config_file["DatasetsSettings"]["artists_info_dataset_filename"] = str(os.path.join(BASEPATH, "datasets",config_file["DatasetsSettings"]["artists_info_dataset_filename"]))
    config_file["DatasetsSettings"]["questions_filename"] = str(os.path.join(BASEPATH, "datasets",config_file["DatasetsSettings"]["questions_filename"]))
    config_file["DatasetsSettings"]["predicted_answers_filename"] = str(os.path.join(BASEPATH, "datasets",config_file["DatasetsSettings"]["predicted_answers_filename"]))
    
    
    return config_file

def main():
    config = load_config_file()
    wiki_loader = WikiDataLoader(config)
    wiki_loader.create_dataset()
    
    question_generator = QuestionGenerator(config)
    question_generator.generate_questions()
    
    question_answering = QuestionsAnswering(config)
    question_answering.generate_answers()

if __name__ == "__main__":
    main()