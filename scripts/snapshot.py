"""Renders the search panel for a query to a PNG, without the hotkey or tray: for checking the UI and for docs.

    python scripts/snapshot.py "a dog on the beach" out.png [--filter N] [--row N]
"""

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from digup.app import theme
from digup.app.panel import SearchPanel
from digup.app.query import QueryService
from digup.settings import Settings


def main() -> None:
    query, out = sys.argv[1], sys.argv[2]
    row = int(sys.argv[sys.argv.index("--row") + 1]) if "--row" in sys.argv else 0
    app = QApplication(sys.argv)
    app.setStyleSheet(theme.stylesheet(theme.palette()))
    service = QueryService()
    panel = SearchPanel(service, Settings.load())
    panel.show_panel()
    panel.search.setText(query)
    finals = []
    service.results.connect(lambda r: r.final and finals.append(r))

    def wait():
        if not finals:
            QTimer.singleShot(200, wait)
            return
        if row:
            panel._move(row)
        QTimer.singleShot(2500, shoot)

    def shoot():
        panel.grab().save(out)
        service.close()
        app.quit()

    QTimer.singleShot(400, wait)
    app.exec()


if __name__ == "__main__":
    main()
