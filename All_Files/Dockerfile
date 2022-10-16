FROM python:3
WORKDIR /app
COPY . .
RUN pip3 install --no-cache-dir -r /app/requirements.txt
EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]

#ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]
# docker build . -f Dockerfile -t walmart_sales_prediction 
# docker run -p 8501:8501 walmart_sales_prediction