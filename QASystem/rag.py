from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings


class myRAG():
    """Класс myRAG используется для создания RAG системы


        Methods
        -------
        make_llm_initialization()
            инициализирует llms
        make_query_pipeline_initialization()
            инициализирует query pipeline
        make_query_chain_initialization()
            инициализирует query chain
        get_response()
            Метод для получения ответа от rag системы.
        clean()
            Этот метод удаляет llm, query_pipeline, qa_chain
        """

    def __init__(self):
        self.make_llm_initialization()
        self.make_query_pipeline_initialization()

    def make_llm_initialization(self, ):
        """ Инициализация llm
        """
        self.model_name = "intfloat/multilingual-e5-large"

        self.model = LlamaCpp(
            model_path="../data/openchat_3.5.Q4_0.gguf",
            temperature=0.6,
            repetition_penalty=1.5,
            n_gpu_layers=0,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            verbose=False,
        )

    def make_query_pipeline_initialization(self):
        """Инициализация query pipeline
        """

        model_name = "intfloat/multilingual-e5-large"
        model_kwargs = {"device": "cpu"}
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        prompt_template = """Ты — автоматизированный виртуальный помощник техподдержки клиентов.
        Используй контекст ниже, чтобы ответить на вопрос в конце и дать краткий ответ.
        Если ты не знаешь ответа, просто отвечай, что не знаешь.

        Контекст: {context}

        Вопрос: {question}

        Ответ:
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=['context', 'question']
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=self.model,
            chain_type='stuff',
            retriever=faiss_index.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={'prompt': PROMPT}
        )

    def get_response(self, query):
        """Метод для получения ответа от rag системы.
        Возвращает словарь из запроса, найденных документов, ответа.

        Parameters
        ----------
        query: str
            запрос системе
        """

        answer = self.qa({"query": query})

        return answer['result']

    def clean(self):
        """Этот метод удаляет model, qa.

        """
        del self.model, self.qa


if __name__ == '__main__':
    query = "Как открыть ПВЗ?"
    rag = myRAG()
    print(rag.get_response(query))
