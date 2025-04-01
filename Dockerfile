FROM python:3.11
WORKDIR /usr/local/app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_ENV=production
ENV FLASK_APP=app.py

# Copy in the source code
COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app"]