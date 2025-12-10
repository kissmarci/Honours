# Memorisation/utils/logging_utils.py
import sys
import re
import logging
import atexit
from datetime import datetime
from pathlib import Path

class RunLogger:
    """
    Manage structured logging + a 'prints' capture file. 
    - structured logs -> run_YYYYMMDD_HHMMSS.log
    - cleaned prints  -> prints_YYYYMMDD_HHMMSS.log
    Options:
    - filter_tqdm: keep tqdm output in the console, don't save its \r-updates to file
    - filter_epoch: keep "Epoch: X/Y" in console, don't save to file
    """

    ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    EPOCH_RE = re.compile(r'^\s*Epoch\s*:?\\s*\\d+\\s*/\\s*\\d+', re.IGNORECASE)

    def __init__(self, base_dir: Path, *,
                 filter_tqdm: bool = True,
                 filter_epoch: bool = True,
                 level: int = logging.INFO):
        self.base_dir = Path(base_dir)
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_path = self.logs_dir / f"run_{ts}.log"
        self.prints_path = self.logs_dir / f"prints_{ts}.log"

        self.filter_tqdm = filter_tqdm
        self.filter_epoch = filter_epoch
        self.level = level

        self._prints_fobj = None
        self._original_stdout = sys.__stdout__
        self._original_stderr = sys.__stderr__
        self._installed = False

    class _Tee:
        def __init__(self, originals, fileobj, ansi_re, epoch_re, filter_tqdm, filter_epoch):
            self.originals = originals  # tuple(sys.__stdout__, sys.__stderr__)
            self.fileobj = fileobj
            self.ansi_re = ansi_re
            self.epoch_re = epoch_re
            self.filter_tqdm = filter_tqdm
            self.filter_epoch = filter_epoch

        def write(self, data):
            if not data:
                return

            # Tqdm-style partial updates: contain '\r' and lack '\n'
            if self.filter_tqdm and '\r' in data and '\n' not in data:
                # show on console(s) only
                for o in self.originals:
                    try:
                        o.write(data)
                        o.flush()
                    except Exception:
                        pass
                return

            # Epoch line pattern, show on console but don't save
            if self.filter_epoch and self.epoch_re.search(data):
                for o in self.originals:
                    try:
                        o.write(data)
                        o.flush()
                    except Exception:
                        pass
                return

            # Strip ANSI for file, keep original for console
            cleaned = self.ansi_re.sub('', data)

            # Avoid writing empty/whitespace-only lines to file (often results after ANSI removal)
            if cleaned.strip() == '':
                for o in self.originals:
                    try:
                        o.write(data)
                        o.flush()
                    except Exception:
                        pass
                return

            # Write to console(s) with original ANSI, to file with cleaned text
            for o in self.originals:
                try:
                    o.write(data)
                    o.flush()
                except Exception:
                    pass

            try:
                self.fileobj.write(cleaned)
                self.fileobj.flush()
            except Exception:
                pass

        def flush(self):
            for o in self.originals:
                try:
                    o.flush()
                except Exception:
                    pass
            try:
                self.fileobj.flush()
            except Exception:
                pass

    def start(self):
        """Install logging handlers and redirect sys.stdout / sys.stderr to the Tee."""
        if self._installed:
            return

        # Structured logger -> both file and console
        logging.basicConfig(
            level=self.level,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[logging.FileHandler(self.run_path), logging.StreamHandler(self._original_stdout)],
        )

        # Open prints file for cleaned print() capture
        self._prints_fobj = open(self.prints_path, "w", encoding="utf-8")

        # Install tee to sys.stdout and sys.stderr
        tee = self._Tee(
            originals=(self._original_stdout, self._original_stderr),
            fileobj=self._prints_fobj,
            ansi_re=self.ANSI_RE,
            epoch_re=self.EPOCH_RE,
            filter_tqdm=self.filter_tqdm,
            filter_epoch=self.filter_epoch,
        )
        sys.stdout = tee
        sys.stderr = tee

        # Make sure we close on exit
        atexit.register(self.close)
        self._installed = True

        logging.info(f"RunLogger started. run_log={self.run_path}, prints_log={self.prints_path}")

    def close(self):
        """Restore stdout/stderr and close files."""
        if not self._installed:
            return
        try:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
        except Exception:
            pass

        try:
            if self._prints_fobj:
                self._prints_fobj.flush()
                self._prints_fobj.close()
        except Exception:
            pass

        self._installed = False
        logging.info("RunLogger closed.")