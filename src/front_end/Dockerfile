FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Create the upload directory that will be used by the volume.
# This is still done as root.
RUN mkdir -p /app/uploads

# Copy requirements file and install dependencies AS ROOT.
# This is the correct place to install packages, before switching users.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy the rest of your application code
COPY . .

# Change ownership of the entire app directory, INCLUDING the new 'uploads' folder
# and all the installed packages, to the non-root user.
RUN chown -R appuser:appgroup /app

# Now, switch to the non-root user for all subsequent commands.
# The application will run as this user.
USER appuser

# Expose the port the app runs on
EXPOSE 8050

# Set the command to run the application
CMD ["sh", "-c", "chown -R appuser:appgroup /app/uploads && exec gunicorn --bind 0.0.0.0:8050 --user appuser --group appgroup --timeout 300 --reload app:server"]

