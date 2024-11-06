# heart-disease-predictor

# Project Setup and Docker Usage

## Prerequisites

Make sure you have the following installed on your machine:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Installation on Windows

### Step 1: Install Docker Desktop

1. Open the downloaded installer file (Docker Desktop Installer.exe).
2. Follow the installer instructions:
    - Accept the license agreement.
    - Choose any additional options (e.g., integration with WSL 2, if available).

3. Click "Install" and wait for the installation to complete.
4. You may need to restart your computer after installation.

### Step 2: Launch Docker Desktop
1. After restarting, Docker Desktop should launch automatically.
2. If not, open Docker Desktop manually from the Start menu.
3. Wait for Docker to initialize. You should see the Docker icon appear in the taskbar.


### Step 3: Configure the `docker-compose.yml`
Ensure the main project directory contains a `docker-compose.yml`

### Step 4: Start the Containers
1. Open PowerShell or Command Prompt.
2. Navigate to the project directory:
```bash
cd path/to/project
```
3. Build image.
```bash
docker-compose build 
```
4. Run the containers with Docker Compose:
```bash
docker-compose up -d
```
The -d option runs the containers in detached mode.

### Step 5: Check container status
To check if the containers are running properly, use the command:

```bash
docker ps
```

### Step 6: Access the Application
Open your web browser and go to:

```
localhost:8080
```

## Installation on Linux (Ubuntu)

### Prerequisites

- **Docker**, **docker-compose** installed on Ubuntu.
- **Project repository** cloned to your local machine.

### Step 1: Manage Docker as a Non-Root User
If you want to avoid using sudo with Docker commands, add your user to the Docker group:

```bash
sudo usermod -aG docker $USER
```
Log out and back in for this to take effect, or use:
```bash
newgrp docker
```

### Step 2: Configure the `docker-compose.yml`
Ensure the main project directory contains a `docker-compose.yml`.

### Step 3: Start the Containers
1. Open Terminal.
2. Navigate to the project directory:
```bash
cd path/to/project
```
3. Run the containers with Docker Compose:
```bash
docker-compose up -d
```
The -d option runs the containers in detached mode.

### Step 4: Check Container Status
To check if the containers are running properly, use the command:
```bash
docker-compose ps
```

### Step 5: Access the Application
Open your web browser and go to:
```
localhost:8080
```

### Step 6: Stop the Containers
To stop the running containers, execute:
```bash
docker-compose down
```

npm install cors
