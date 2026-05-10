import sys
import re

from PySide6.QtCore import (
    QThread,
    Signal,
)

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
)

from inference import run_inference


class Worker(QThread):

    finished = Signal(list)

    def __init__(self, tic_ids):

        super().__init__()

        self.tic_ids = tic_ids

    def run(self):

        results = []

        for tic in self.tic_ids:

            try:

                result = run_inference(tic)

                results.append(result)

            except Exception as exc:

                results.append({
                    "tic_id": tic,
                    "probability": 0.0,
                    "candidate": False,
                    "snr": 0.0,
                    "error": str(exc),
                })

        results.sort(
            key=lambda x: x["probability"],
            reverse=True
        )

        self.finished.emit(results)


class ExoplanetApp(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Exoplanet Candidate Finder"
        )

        self.resize(900, 600)

        layout = QVBoxLayout()

        self.label = QLabel(
            "Enter TIC IDs"
        )

        layout.addWidget(self.label)

        self.input_box = QTextEdit()

        layout.addWidget(self.input_box)

        self.button = QPushButton(
            "Analyze"
        )

        self.button.clicked.connect(
            self.analyze
        )

        layout.addWidget(self.button)

        self.status = QLabel("Ready")

        layout.addWidget(self.status)

        self.table = QTableWidget()

        self.table.setColumnCount(4)

        self.table.setHorizontalHeaderLabels([
            "TIC ID",
            "Probability",
            "Candidate",
            "SNR",
        ])

        layout.addWidget(self.table)

        self.setLayout(layout)

    def analyze(self):

        text = self.input_box.toPlainText()

        lines = re.split(r'[,\s]+', text)

        tic_ids = []

        for line in lines:

            line = line.strip()

            if line:

                try:
                    tic_ids.append(int(line))

                except:
                    pass

        if len(tic_ids) == 0:
            return

        self.button.setEnabled(False)

        self.status.setText(
            "Running inference..."
        )

        self.worker = Worker(tic_ids)

        self.worker.finished.connect(
            self.display_results
        )

        self.worker.start()

    def display_results(self, results):

        self.table.setRowCount(len(results))

        for row, result in enumerate(results):

            self.table.setItem(
                row,
                0,
                QTableWidgetItem(
                    str(result["tic_id"])
                )
            )

            self.table.setItem(
                row,
                1,
                QTableWidgetItem(
                    f'{result["probability"]:.4f}'
                )
            )

            self.table.setItem(
                row,
                2,
                QTableWidgetItem(
                    str(result["candidate"])
                )
            )

            self.table.setItem(
                row,
                3,
                QTableWidgetItem(
                    f'{result["snr"]:.2f}'
                )
            )

        self.status.setText(
            "Finished"
        )

        self.button.setEnabled(True)


app = QApplication(sys.argv)

window = ExoplanetApp()

window.show()

sys.exit(app.exec())