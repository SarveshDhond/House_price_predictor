# PYTHON IMAGE
FROM python:3.11

# SET WORKING DIRECTORY FOR CLIENT NOT FOR US
WORKDIR /app

# COPY AND RUN DEPENDENCIES
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt # ONLY NAME NO VERSIONS

# COPY FASTAPI CODE AND MODEL
COPY ./app ./app
COPY ./model ./model

# PORT
EXPOSE 8000

# RUN FASTAPI
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]