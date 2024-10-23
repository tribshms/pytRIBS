import docker
import os
import subprocess
import platform
import time


class MeshBuilderDocker:
    """
    A class to manage the execution of the MeshBuilder tool in a Docker container.

    This class facilitates setting up and running the MeshBuilder tool within a Docker container. It allows
    for specifying the Docker image and volume path. The class manages the creation and lifecycle of Docker
    containers for executing the MeshBuilder tool.

    Parameters
    ----------
    volume_path : str
        Path to the directory that will be mounted as a volume inside the Docker container.

    Attributes
    ----------
    volume_path : str
        The path to the directory that is mounted as a volume inside the Docker container.

    """

    def __init__(self, volume_path):
        self.image_name = "tribs/meshbuilder:latest"
        self.volume_path = volume_path
        self.client = None
        self.container = None

    # need to generate input file with POINTFILENAME: That's it.
    # then
    def start_docker_desktop(self, attempts=0):
        """
        Ensure Docker is running, and if not, attempt to start it or prompt for installation.

        This method checks if Docker is running by attempting to ping the Docker client. If Docker is not running,
        it attempts to start Docker based on the current operating system. If Docker cannot be started after a
        specified number of attempts, it prompts for installation.

        Parameters
        ----------
        attempts : int, optional
            The current number of attempts made to start Docker, by default 0.

        Returns
        -------
        bool
            Returns True if Docker is running or successfully started, otherwise False.

        Raises
        ------
        docker.errors.DockerException
            If there is an issue connecting to Docker or starting it.

        Notes
        -----
        - On Windows, the method attempts to start Docker Desktop via PowerShell.
        - On macOS, the method uses the `open` command to launch Docker.
        - On Linux, the method uses `systemctl` to start the Docker service.
        - If the operating system is not recognized, the method prints an error and returns False.
        - The method will attempt to start Docker up to 5 times, waiting 15 seconds between each attempt.
        """
        max_attempts = 5
        system = platform.system()

        try:
            self.client = docker.from_env()
            self.client.ping()
            print("Docker is running.")
            return True
        except docker.errors.DockerException:
            if attempts >= max_attempts:
                print(f"Failed to start Docker after {max_attempts} attempts.")
                self._prompt_docker_installation(system)
                return False

            print(f"Docker is not running. Attempt {attempts + 1}/{max_attempts} to start Docker...")

            if system == 'Windows':
                subprocess.run(["powershell", "-Command", "Start-Process", "Docker Desktop"])
            elif system == 'Darwin':  # macOS
                subprocess.run(["open", "-a", "Docker"])
            elif system == 'Linux':
                subprocess.run(["systemctl", "start", "docker"])
            else:
                print(f"Unsupported system: {system}")
                return False

            time.sleep(15)
            return self.start_docker_desktop(attempts + 1)

    @staticmethod
    def _prompt_docker_installation(system):
        print("Failed to start Docker. Please start Docker manually or install it if not available.")
        if system in ['Windows', 'Darwin']:
            print("Download and install Docker Desktop from https://www.docker.com/products/docker-desktop")
        elif system == 'Linux':
            print("Install Docker using your package manager, e.g., 'sudo apt install docker.io' for Ubuntu.")

    def initialize_docker_client(self):
        """
        Initialize the Docker client.

        This method attempts to connect to the Docker daemon using the Docker client. If the connection is successful,
        it assigns the client to the `self.client` attribute. If the connection fails, it prints an error message and
        raises an exception.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the connection to the Docker daemon fails, the exception is caught and re-raised.

        Notes
        -----
        This method assumes that Docker is properly installed and running on the system.
        """
        try:
            self.client = docker.from_env()
            print("Connected to Docker daemon successfully.")
        except Exception as e:
            print(f"Failed to connect to Docker daemon: {e}")
            raise

    def pull_image(self):
        """
        Pull the Docker image.

        This method pulls the Docker image specified by the `self.image_name` attribute from a Docker registry.
        If the image pull is successful, a success message is printed. If an error occurs during the process,
        it prints an error message and raises the exception.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        Exception
            If there is an error while pulling the Docker image, the exception is caught and re-raised.

        Notes
        -----
        The method expects that the Docker client (`self.client`) has already been initialized and connected to
        the Docker daemon, and that `self.image_name` contains the name of the Docker image to pull.
        """
        try:
            print(f"Pulling the Docker image {self.image_name}...")
            self.client.images.pull(self.image_name)
            print("Image pulled successfully.")
        except Exception as e:
            print(f"Error pulling Docker image: {e}")
            raise

    def run_container(self):
        """
        Run the Docker container with the specified volume.

        This method starts a Docker container using the specified Docker image and mounts a volume from the
        host to the container. The volume is mounted in read-write mode at the `/meshbuild/data` path inside
        the container. The container is run with interactive terminal options (`tty` and `stdin_open`), and
        it is detached from the terminal.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an error occurs while starting the Docker container, the exception is caught and re-raised.

        Notes
        -----
        - On Windows systems, the method replaces backslashes (`\\`) in the volume path with forward slashes (`/`).
        - The Docker client (`self.client`) must be initialized and connected to the Docker daemon before running this method.
        - The volume path (`self.volume_path`) must be correctly set to a valid host directory.
        - The container runs with `/bin/bash` as the entrypoint.
        """
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
        """
        Execute a command in the running Docker container.

        This method runs a specified shell command inside the currently running Docker container. The output
        of the command is streamed and printed line by line. If the command runs successfully, a success message
        is printed; otherwise, the exit code is displayed.

        Parameters
        ----------
        command : str
            The shell command to be executed inside the Docker container.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If there is an error while executing the command inside the Docker container, the exception is caught and re-raised.

        Notes
        -----
        - The container must be running before executing this method.
        - The method uses the `exec_run` function from the Docker client to run the command and stream the output.
        - The command is executed in the `/bin/bash` shell of the container.
        - The exit code of the command is checked, and success or failure is reported accordingly.
        """

        try:
            print(f"Executing command in the container: {command}")
            # Use the shell to run commands
            # exec_command = f"bash -c 'cd /tribs/shared && {command}'"
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

        This method executes a sequence of commands inside the running Docker container to perform the
        MeshBuilder workflow. The workflow involves copying necessary files, running the MeshBuilder tool,
        and executing partitioning using the METIS tool.

        Parameters
        ----------
        file_path : str
            Path to the .in file to be used by MeshBuilder.
        nn : int
            Number of computer nodes for partitioning.
        OPT_Part : int
            Partitioning method to be used.
        basename : str
            Simulation basename for the output files.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an error occurs during the execution of the commands inside the Docker container, the exception is caught and re-raised.

        Notes
        -----
        - The container must be running before this method is executed.
        - The method runs several shell commands in sequence inside the Docker container to set up the environment,
          run MeshBuilder, and partition the mesh.
        - The commands are executed using the `exec_run` function from the Docker client with `sh -c` to ensure proper execution.
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
        """
        Stop and remove the Docker container.

        This method stops the currently running Docker container and then removes it. If an error occurs during
        stopping or removing the container, the exception is caught, and an error message is displayed.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an error occurs while stopping or removing the Docker container, the exception is caught and re-raised.

        Notes
        -----
        - The method assumes that `self.container` is a valid Docker container object that is running.
        """
        try:
            print("Stopping and removing the container...")
            self.container.stop()
            self.container.remove()
            print("Container stopped and removed.")
        except Exception as e:
            print(f"Error stopping or removing the container: {e}")

    def clean_directory(self):
        """
        Clean the directory by removing intermediate files.

        This method removes all files in the specified volume directory, except those with extensions `.in`,
        `.points`, `.reach`, or `.out`. It ensures that only intermediate files are deleted, leaving essential
        files intact. If an error occurs while deleting a file, an error message is printed.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an error occurs while deleting a file, the exception is caught and a message is displayed.

        Notes
        -----
        - The method iterates through the files in the `self.volume_path` directory.
        - Files with the extensions `.in`, `.points`, `.reach`, and `.out` are preserved, while others are deleted.
        """

        directory_path = self.volume_path

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path) and not filename.endswith(('.in', '.points', '.reach', '.out')):
                try:
                    os.remove(file_path)
                    # print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
