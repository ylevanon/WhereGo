
FROM python:latest

# Set the application directory
WORKDIR /WhereGo
#ENV VIRTUAL_ENV=venv
#RUN python3 -m venv ${VIRTUAL_ENV}
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
ADD . /WhereGo


# Execute the code
CMD ["python","-u", "run.py"]ps -e