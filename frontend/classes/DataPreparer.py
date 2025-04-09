import base64
import csv
import json
import os
import shutil  # To move files
import zipfile


class DataPreparer:
    """A class used to prepare the PDF data for download."""

    def __init__(self):
        # Create a temporary directory to store the PDF data
        self.tmp_dir = os.path.join(os.getcwd(), "frontend/temp_pdf_data")
        self.tmp_archive_dir = os.path.join(os.getcwd(), "frontend/archive")
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.tmp_archive_dir, exist_ok=True)

    def _initial(self, upload_file_name: str) -> None:
        """Create initial temp folders for uploaded PDF file."""

        self.tmp_work_dir = os.path.join(self.tmp_dir, upload_file_name)
        os.makedirs(self.tmp_work_dir, exist_ok=True)
        for folder in ["text", "tables", "images"]:
            os.makedirs(os.path.join(self.tmp_work_dir, folder), exist_ok=True)

    def cleanup(self) -> None:
        """Removes temporary files after processing."""

        # Remove zipped files in archive folder
        if os.path.exists(self.tmp_archive_dir):
            for file in os.listdir(self.tmp_archive_dir):
                file_path = os.path.join(self.tmp_archive_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Remove temporary work directory in `temp_pdf_data` folder
        if os.path.isdir(self.tmp_work_dir):
            shutil.rmtree(self.tmp_work_dir, ignore_errors=True)

    def _zip_folder(self, src_folder, out_zip_path) -> None:
        """Create a zip folder for user to download."""

        with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(src_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Write file to zip with relative path to source_folder
                    arcname = os.path.relpath(file_path, start=src_folder)
                    zipf.write(file_path, arcname)

    def prepare_pdf_data(
        self,
        upload_file_name: str,
        pdf_data: list[dict],
        all_text: str,
        all_text_translated: str,
    ) -> str:
        """Prepare the PDF data for download.

        Parameters
        ----------
        uploaded_file_name : str
            The file name. To be used as folder name.
        pdf_data : list[dict]
            The PDF data in session state. Contains the text, images, and tables in all pages.
        all_text : str
            The concatenated string of all text in the PDF.
        all_text_translated : str
            The concatenated string of all translated text in the PDF.

        Returns
        -------
        str
            The path to the prepared zipped PDF data folder.
        """

        # Prepare temporary folders
        upload_file_name = upload_file_name.replace(".pdf", "")
        self._initial(upload_file_name)

        # Save the extracted text
        with open(
            os.path.join(self.tmp_work_dir, "text", "all_text.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(all_text)

        # Save the translated text
        with open(
            os.path.join(self.tmp_work_dir, "text", "all_text_translated.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(all_text_translated)

        # Save the tables and images
        for page_num, page in enumerate(pdf_data):
            # Tables
            for table_idx, table in enumerate(page.get("tables", [])):
                table_path = os.path.join(
                    self.tmp_work_dir,
                    "tables",
                    f"table_{page_num + 1}_{table_idx + 1}.csv",
                )
                with open(table_path, "w") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(table)

            # Images
            for img_idx, image in enumerate(page.get("images", [])):
                img_file_path = os.path.join(
                    self.tmp_work_dir, "images", image.get("img_filename")
                )
                img_b64 = image.get("img_b64")

                if img_b64:
                    image_data = base64.b64decode(img_b64)
                    with open(img_file_path, "wb") as f:
                        f.write(image_data)
                else:
                    continue

        # Before saving the PDF data, exclude img_b64 and img_filepath
        clean_pdf_data = []
        for page in pdf_data:
            clean_page = dict(page)
            if "images" in page:
                clean_page["images"] = []
                for image in page["images"]:
                    # Create a new image dict excluding the unwanted keys
                    clean_image = {
                        k: v
                        for k, v in image.items()
                        if k not in ("image_url", "img_b64")
                    }
                    clean_page["images"].append(clean_image)
            clean_pdf_data.append(clean_page)

        # Save the PDF data
        with open(os.path.join(self.tmp_work_dir, "pdf_data.json"), "w") as f:
            f.write(json.dumps(clean_pdf_data, indent=4))

        # Zip the files in a folder
        self._zip_folder(
            self.tmp_work_dir,
            os.path.join(self.tmp_archive_dir, f"{upload_file_name}.zip"),
        )
        return os.path.join(self.tmp_archive_dir, upload_file_name)
