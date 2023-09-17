FROM python:3.9
EXPOSE 8080
WORKDIR /app

COPY requirements.txt ./
COPY install-treesitter.py ./
COPY tree-sitter-python/ ./tree-sitter-python/
COPY llm.py ./
COPY app.py ./

COPY data ./data

RUN pip install tree-sitter
RUN python install-treesitter.py
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]