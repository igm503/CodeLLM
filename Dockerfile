FROM python:3.9
EXPOSE 8080
WORKDIR /app
COPY . ./
RUN pip install -r requirements.txt
RUN python install-treesitter.py
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]