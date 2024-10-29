# heart-disease-predictor

# Project Setup and Docker Usage

## Prerequisites

Make sure you have the following installed on your machine:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Building Docker Images

To build the Docker images for all services (frontend, backend, and AI module), navigate to the root directory of the project and run:

```bash
docker-compose build
```

This command reads the docker-compose.yml file and builds the images according to the specifications provided.

## Running the Containers

Once the images are built, you can start the containers using:

```bash
docker-compose up
```

This command will start all services defined in the docker-compose.yml. You can access:

    Frontend: http://localhost:8080
    Backend: http://localhost:3000
    AI Module: http://localhost:5000

To run the containers in detached mode (background), use:

```bash
docker-compose up -d
```

Use this:
```bash
docker ps
```
to check which containers are active.

# Viewing Changes

## During Development
While developing, you may want to see changes reflected in the running containers without rebuilding the images each time. Here’s how to do it:

### Frontend Changes:
Navigate to the frontend directory.
Run your development server:

```bash
npm run serve
```

This allows you to see changes in real-time in your browser. However, note that you won't be using Docker for this specific part.

### Backend Changes:
For the Node.js backend, if you are not in Docker, run:

```bash
npm run start
```

If you want to see changes in Docker, you will need to rebuild the backend service using:

```bash
docker-compose up --build backend
```

### AI Module Changes:
Similar to the backend, you will need to rebuild the AI module if you make changes to the code. Use:

```bash
docker-compose up --build ai_module
```

## After Making Changes
If you make changes to any service (frontend, backend, or AI module), you will need to rebuild the relevant Docker image and restart the container. You can do this for all services with:

```bash
docker-compose up --build
```

## Stopping the Containers
To stop the running containers, use:

```bash
docker-compose down
```

This command stops and removes the containers, but retains the built images. If you want to remove the images as well, use:

```bash
docker-compose down --rmi all
```

npm install cors
