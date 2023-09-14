from NLP import API, ML
from uuid import uuid4
import os



class Environment:
    def __init__(self, api_key) -> None:
        # file = open("db.txt", "w+")
        self.api = API(api_key)
        self.ml = ML()


    def create_user(self, username, password):
        # create user id
        user_id = str(uuid4())
        # add user to db
        with open('db.txt', 'a') as db:
            db.write("{},{},{}\n".format(username, password, user_id))
        os.mkdir('./DB/'+user_id)
        self.ml.create_model(user_id)
        # init neural net

    def login(self, username, password):
        with open('db.txt', 'r') as db:
            lines = db.readlines()
            for line in lines:
                if username in line and password in line:
                    user_id = line.split(',')[2].strip()
                    self.ml.start_session(user_id)
                    return user_id
        return None

    def logout(self, user_id):
        self.ml.close_session(user_id)

    def list_user_docs(self, user_id):
        directory_list = []
        for root, dirs, files in os.walk("./DB/"+user_id, topdown=False):
            for name in dirs:
                directory_list.append(os.path.join(root, name))

        return directory_list

    
    def upload_doc(self, user_id, doc_name):
        doc_path = "./DB/"+user_id+"/"+doc_name+"/"
        # os.mkdir(doc_path)
        doc_text = open(doc_path+"doc.txt", "r").read()
        # open(doc_path+"original.txt", "w").write(doc_text)
        # analyze the doc right away using the ML
        doc_context = self.api.extract_doc_context(doc_text)
        open(doc_path+"context.txt", "w").write(doc_context)
        open(doc_path+"aug_doc.txt", "w").write(doc_text)

        # generate the requests using the ML model, and carry out the operations straight away

    def make_request(self, user_id, doc_name, request):
        doc_path = "./DB/"+user_id+"/"+doc_name+"/"
        doc_context = open(doc_path+"context.txt", "r").read()
        request_operation = self.api.extract_request_operation(doc_context, request)
        # print(request, request_operation)
        if request_operation.lower().find("answer")!=-1:
            answer = self.api.answer_question(doc_context, request)
            self.ml.train_model(doc_context, request)
            return answer
        else:
            doc_text = open(doc_path+"aug_doc.txt", "r").read()
            augmented_doc = self.api.augment_doc(doc_text, doc_context, request, request_operation)
            open(doc_path+"aug_doc.txt", "w").write(augmented_doc)
            return augmented_doc


    def retrieve_doc(self, user_id, doc_name):
        doc_path = "./DB"+user_id+"/"+doc_name
        return open(doc_path+"doc.txt", "r").read(), open(doc_path+"aug_doc.txt", "r").read(), open(doc_path+"context.txt", "r").read()

