import importlib.metadata
import logging
import os
import subprocess
import sys
from urllib.parse import urlparse



def install_and_import(package, version="", params="", link="", packageimportname=""):
    try:
        if importlib.metadata.version(package) != version:
            raise ImportError
        importlib.import_module(package)
    except ImportError:
        pass

        installation_str = package
        installation_cmd_list = ["install"]

        if version:
            installation_str += "==" + version
        installation_cmd_list.append(installation_str)

        if params:
            installation_cmd_list.append(params)

        if link:
            installation_cmd_list.append(link)
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        except Exception as e:
            print(e)

        try:
            if "INSTALL_IGNORE_SSL" in os.environ and os.environ["INSTALL_IGNORE_SSL"] == True:
                domain = urlparse(link).netloc
                installation_cmd_list.append("--trusted-host")
                installation_cmd_list.append(domain)
                installation_cmd_list.append("--trusted-host")
                installation_cmd_list.append("pypi.org")
                installation_cmd_list.append("--trusted-host")
                installation_cmd_list.append("files.pythonhosted.org")
            subprocess.check_call([sys.executable, "-m", "pip", *installation_cmd_list])
        except Exception as e:
            print(e)
    finally:
        if not packageimportname:
            globals()[package] = importlib.import_module(package)
        else:
            globals()[packageimportname] = importlib.import_module(packageimportname)


def execute_bash_command(cmd):
    tenv = os.environ.copy()
    tenv["LC_ALL"] = "C"
    bash_command = cmd
    process = subprocess.Popen(
        bash_command.split(), stdout=subprocess.PIPE, env=tenv
    )
    return process.communicate()[0]


def check_gpu_and_torch_compatibility():
    try:
        import platform

        if platform.system() == "Windows":
            install_and_import(
                "torch",
                "1.12.1+cu116",
                "-f",
                "https://download.pytorch.org/whl/torch_stable.html",
            )
        else:
            bash_command = "nvidia-smi --query-gpu=name --format=csv"
            output = ""
            try:
                output = execute_bash_command(bash_command).decode()
            except Exception as e:
                try:
                    import torch
                except Exception as e:
                    install_and_import("torch")


            if "NVIDIA A100" in output:
                install_and_import(
                    "torch",
                    "1.11.0+cu113",
                    "-f",
                    "https://download.pytorch.org/whl/torch_stable.html",
                )
                install_and_import(
                    "torchvision",
                    "0.12.0+cu113",
                    "-f",
                    "https://download.pytorch.org/whl/torch_stable.html",
                )
            else:
                try:
                    import torch
                except Exception as e:
                    install_and_import("torch")
    except OSError as e:
        logging.info("GPU device is not available")
