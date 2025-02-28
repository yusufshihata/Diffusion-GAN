import os
import json

class Logger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_log.json")
        self.logs = {"generator_loss": [], "discriminator_loss": []}

    @classmethod
    def log(self, epoch, gen_loss, disc_loss):
        self.logs["generator_loss"].append({"epoch": epoch, "loss": gen_loss})
        self.logs["discriminator_loss"].append({"epoch": epoch, "loss": disc_loss})
        
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=4)
