import docker
import os
import subprocess
import platform
import time

class RunTribsDocker:
    def __init__(self, volume_path):
        self.image_name = "tribs/tribs:latest"
        self.volume_path = volume_path
        self.client = None
        self.container = None
    
    def start_docker_desktop(self):
        """Ensure Docker is running, and if not, start it or prompt for installation."""
        system = platform.system()

        try:
            # Check if Docker is running
            self.client = docker.from_env()
            self.client.ping()
            print("Docker is running.")
        except docker.errors.DockerException:
            print("Docker is not running. Attempting to start Docker...")

            if system == 'Windows':
                # Start Docker on Windows
                subprocess.run(["powershell", "-Command", "Start-Process", "Docker Desktop"])
            elif system == 'Darwin':  # macOS
                # Start Docker on macOS
                subprocess.run(["open", "-a", "Docker"])
            elif system == 'Linux':
                # Start Docker on Linux
                subprocess.run(["systemctl", "start", "docker"])
            else:
                print(f"Unsupported system: {system}")
                return

            # Wait a bit for Docker to start
            time.sleep(10)

            # Check again if Docker is running
            try:
                self.client.ping()
                print("Docker started successfully.")
            except docker.errors.DockerException:
                print("Failed to start Docker. Please start Docker manually or install it if not available.")
                if system == 'Windows':
                    print("Download and install Docker Desktop from https://www.docker.com/products/docker-desktop")
                elif system == 'Darwin':
                    print("Download and install Docker Desktop from https://www.docker.com/products/docker-desktop")
                elif system == 'Linux':
                    print("Install Docker using your package manager, e.g., 'sudo apt install docker.io' for Ubuntu.")
    
    def initialize_docker_client(self):
        """Initialize the Docker client."""
        try:
            self.client = docker.from_env()
            print("Connected to Docker daemon successfully.")
        except Exception as e:
            print(f"Failed to connect to Docker daemon: {e}")
            raise
    
    def pull_image(self):
        """Pull the Docker image."""
        try:
            print(f"Pulling the Docker image {self.image_name}...")
            self.client.images.pull(self.image_name)
            print("Image pulled successfully.")
        except Exception as e:
            print(f"Error pulling Docker image: {e}")
            raise
    
    def run_container(self):
        """Run the Docker container with the specified volume."""
        if os.name == 'nt':
            self.volume_path = self.volume_path.replace('\\', '/')

        try:
            print("Running the Docker container...")
            self.container = self.client.containers.run(
                self.image_name,
                entrypoint="/bin/bash",
                volumes={self.volume_path: {'bind': '/tribs/shared', 'mode': 'rw'}},
                tty=True,
                stdin_open=True,
                detach=True
            )
            print("Container started successfully.")
        except Exception as e:
            print(f"Error running Docker container: {e}")
            raise
    
    def execute_command_in_container(self, command):
        """Execute a command in the running Docker container."""
        try:
            print(f"Executing command in the container: {command}")
            # Use the shell to run commands
            exec_command = f"bash -c 'cd /tribs/shared && {command}'"
            exit_code, output = self.container.exec_run(exec_command, tty=True, stream=True)
            print("Command executed. Output:")
            for line in output:
                print(line.decode('utf-8').strip())

            if exit_code == 0:
                print("Command ran successfully.")
            else:
                print(f"Command failed with exit code {exit_code}.")
        except Exception as e:
            print(f"Error running commands in the container: {e}")
    
    def cleanup_container(self):
        """Stop and remove the Docker container."""
        try:
            print("Stopping and removing the container...")
            self.container.stop()
            self.container.remove()
            print("Container stopped and removed.")
        except Exception as e:
            print(f"Error stopping or removing the container: {e}")

    def run_serial(self):
        """Run the serial version of tRIBS."""
        command = "/tribs/bin/tRIBS src/in_files/big_spring.in"
        self.execute_command_in_container(command)

    def run_parallel(self, num_processes):
        """Run the parallel version of tRIBS with MPI."""
        if num_processes < 1:
            print("Error: The number of processes must be 1 or more.")
            return
        command = f"mpirun -np {num_processes} /tribs/bin/tRIBSpar src/in_files/big_spring.in"
        self.execute_command_in_container(command)

    def choose_execution_mode(self):
        """Prompt the user to choose between serial and parallel execution."""
        mode = input("Do you want to run the serial version of tRIBS? (yes/no): ").strip().lower()
        if mode == 'yes':
            print("Running the serial version of tRIBS.")
            self.run_serial()
        elif mode == 'no':
            print("Running the parallel version of tRIBS.")
            num_processes = int(input("Enter the number of processes for parallel execution (1 or more): ").strip())
            self.run_parallel(num_processes)
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
            self.choose_execution_mode()

