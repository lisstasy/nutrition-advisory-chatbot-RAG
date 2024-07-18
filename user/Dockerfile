FROM python:3.8
EXPOSE 8084
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./
ENTRYPOINT ["streamlit", "run", "user.py", "--server.port=8084", "--server.address=0.0.0.0"]