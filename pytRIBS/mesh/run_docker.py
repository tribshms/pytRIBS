import docker
import os
import subprocess
import platform
import time

class MeshBuilderDocker:
    def __init__(self, volume_path):
        self.image_name = "tribs/meshbuilder:latest"
        self.volume_path = volume_path
        self.client = None
        self.container = None

    # need to generate input file with POINTFILENAME: That's it.
    # then
    def start_docker_desktop(self):
        """Ensure Docker is running, and if not, start it or prompt for installation."""
        system = platform.system()

        try:
            self.client = docker.from_env()
            self.client.ping()
        except docker.errors.DockerException:
            print("Docker is not running. Attempting to start Docker...")

            if system == 'Windows':
                subprocess.run(["powershell", "-Command", "Start-Process", "Docker Desktop"])
            elif system == 'Darwin':  # macOS
                subprocess.run(["open", "-a", "Docker"])
            elif system == 'Linux':
                subprocess.run(["systemctl", "start", "docker"])
            else:
                print(f"Unsupported system: {system}")
                return
            time.sleep(10)

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
                volumes={self.volume_path: {'bind': '/meshbuild/data', 'mode': 'rw'}},
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
            #exec_command = f"bash -c 'cd /tribs/shared && {command}'"
            exit_code, output = self.container.exec_run(command, tty=True, stream=True)
            print("Command executed. Output:")
            for line in output:
                print(line.decode('utf-8').strip())

            if exit_code == 0:
                print("Command ran successfully.")
            else:
                print(f"Command failed with exit code {exit_code}.")
        except Exception as e:
            print(f"Error running commands in the container: {e}")

    def execute_meshbuild_workflow(self, file_path, nn, OPT_Part, basename):
        """
        Execute the MeshBuilder workflow directly in the running Docker container.

        :param file_path: Path to the .in file to be used by MeshBuilder.
        :param nn: Number of computer nodes for partitioning (should be an integer).
        :param OPT_Part: Partitioning method.
        :param basename: Simulation basename.
        """
        try:
            # Convert nn to string to ensure it is passed correctly
            nn_str = str(nn)
            OPT_Part_str = str(OPT_Part)

            # Define the commands to be run in sequence
            commands = [
                "cp src/workflow/* data/",
                "cp build/MeshBuilder data/",
                "cp src/metis_builds/METIS/build/programs/gpmetis data/",
                f"cd data && ./MeshBuilder {file_path} && ./run_metis.zsh {nn_str} {OPT_Part_str} {basename}"
            ]

            for command in commands:
                exit_code, output = self.container.exec_run(f"sh -c '{command}'", tty=True, stream=True)
                for line in output:
                    print(line.decode('utf-8').strip())
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

    def clean_directory(self):
        directory_path = self.volume_path

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path) and not filename.endswith(('.in', '.points', '.reach')):
                try:
                    os.remove(file_path)
                    #print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
