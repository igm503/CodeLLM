FROM python:3.10
EXPOSE 8080
WORKDIR /app

COPY requirements.txt ./
COPY tree-sitter/ ./tree-sitter
COPY models/ ./models
COPY app.py ./

RUN pip install tree-sitter
RUN python tree-sitter/install-treesitter.py
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
