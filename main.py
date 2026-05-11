import sys
import re
import csv

from PySide6.QtCore import (
    QThread,
    Signal,
)

from PySide6.QtGui import (
    QIcon,
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
    QFileDialog,
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

                print(
                    f"ERROR FOR TIC {tic}: {exc}"
                )

                results.append({
                    "tic_id": tic,
                    "probability": -1.0,
                    "candidate": False,
                    "snr": 0.0,
                    "period": 0.0,
                    "bls_power": 0.0,
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

        self.results = []

        self.setWindowTitle(
            "TransitAI - Exoplanet Candidate Finder"
        )

        self.setWindowIcon(
            QIcon("icon.ico")
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

        self.export_button = QPushButton(
            "Export CSV"
        )

        self.export_button.clicked.connect(
            self.export_csv
        )

        self.export_button.setEnabled(False)

        layout.addWidget(self.export_button)

        self.status = QLabel("Ready")

        layout.addWidget(self.status)

        self.table = QTableWidget()

        self.table.setColumnCount(6)

        self.table.setHorizontalHeaderLabels([
            "TIC ID",
            "Probability",
            "Candidate",
            "SNR",
            "Period",
            "BLS Power",
        ])

        layout.addWidget(self.table)

        self.footer = QLabel(
            "© 2026 Aashay Kadu"
        )

        layout.addWidget(self.footer)

        self.setLayout(layout)

    def analyze(self):

        text = self.input_box.toPlainText()

        lines = re.split(
            r'[,\s]+',
            text
        )

        tic_ids = []

        for line in lines:

            line = line.strip()

            if line:

                try:

                    tic_ids.append(
                        int(line)
                    )

                except:
                    pass

        if len(tic_ids) == 0:

            self.status.setText(
                "No valid TIC IDs"
            )

            return

        self.button.setEnabled(False)

        self.export_button.setEnabled(False)

        self.status.setText(
            "Running inference..."
        )

        self.worker = Worker(tic_ids)

        self.worker.finished.connect(
            self.display_results
        )

        self.worker.start()

    def display_results(self, results):

        self.results = results

        self.table.setRowCount(
            len(results)
        )

        for row, result in enumerate(results):

            self.table.setItem(
                row,
                0,
                QTableWidgetItem(
                    str(
                        result.get(
                            "tic_id",
                            ""
                        )
                    )
                )
            )

            probability = result.get(
                "probability",
                -1.0
            )

            if probability < 0:

                probability_text = "ERROR"

            else:

                probability_text = (
                    f"{probability:.4f}"
                )

            self.table.setItem(
                row,
                1,
                QTableWidgetItem(
                    probability_text
                )
            )

            self.table.setItem(
                row,
                2,
                QTableWidgetItem(
                    str(
                        result.get(
                            "candidate",
                            False
                        )
                    )
                )
            )

            self.table.setItem(
                row,
                3,
                QTableWidgetItem(
                    f'{result.get("snr", 0.0):.2f}'
                )
            )

            self.table.setItem(
                row,
                4,
                QTableWidgetItem(
                    f'{result.get("period", 0.0):.4f}'
                )
            )

            self.table.setItem(
                row,
                5,
                QTableWidgetItem(
                    f'{result.get("bls_power", 0.0):.2f}'
                )
            )

        self.status.setText(
            "Finished"
        )

        self.button.setEnabled(True)

        self.export_button.setEnabled(True)

    def export_csv(self):

        if len(self.results) == 0:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "exoplanet_candidates.csv",
            "CSV Files (*.csv)"
        )

        if not path:
            return

        with open(
            path,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "tic_id",
                "probability",
                "candidate",
                "snr",
                "period",
                "bls_power",
            ])

            for result in self.results:

                writer.writerow([
                    result.get(
                        "tic_id",
                        ""
                    ),
                    result.get(
                        "probability",
                        -1.0
                    ),
                    result.get(
                        "candidate",
                        False
                    ),
                    result.get(
                        "snr",
                        0.0
                    ),
                    result.get(
                        "period",
                        0.0
                    ),
                    result.get(
                        "bls_power",
                        0.0
                    ),
                ])

        self.status.setText(
            f"Saved CSV: {path}"
        )


QApplication.setApplicationName(
    "TransitAI"
)

QApplication.setOrganizationName(
    "Aashay Kadu"
)

app = QApplication(sys.argv)

window = ExoplanetApp()

window.show()

sys.exit(app.exec())