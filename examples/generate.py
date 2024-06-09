import os
import random
import pandas as pd
from fpdf import FPDF
from pdf2image import convert_from_path
from multiprocessing import Pool, cpu_count
import names

# Templates for each class
templates = {
    "Cancelation": """\
    Cancelation Request
    -------------------
    PRENAME: {prename}
    SURNAME: {surname}
    DATE: {date}

    I would like to cancel my subscription/service.
    """,
    "IBAN Change": """\
    IBAN Change Request
    -------------------
    PRENAME: {prename}
    SURNAME: {surname}
    IBAN: {iban}
    DATE: {date}

    I would like to change my IBAN to the above number.
    """,
    "Damage Report": """\
    Damage Report
    -------------
    PRENAME: {prename}
    SURNAME: {surname}
    CAR NUMBER: {car_number}
    DATE: {date}

    I would like to report damage to my vehicle.
    """
}
first_names = [names.get_first_name() for _ in range(50)]
last_names = [names.get_last_name() for _ in range(50)]
ibans = [f"DE{random.randint(10000000000000000000, 99999999999999999999)}" for _ in range(50)]
car_numbers = [f"{random.choice(['ABC', 'XYZ', 'LMN', 'DEF'])}{random.randint(1000, 9999)}" for _ in range(50)]
dates = [f"{random.randint(2022, 2024)}-{str(random.randint(1, 12)).zfill(2)}-{str(random.randint(1, 28)).zfill(2)}" for _ in range(50)]


# Directory to save generated files
output_dir = "./dataframe_data"
os.makedirs(output_dir, exist_ok=True)

def generate_document(template_name):
    """
    Generates a document based on the given template name by filling it with random sample data.

    Args:
        template_name (str): The name of the template to use for generating the document.

    Returns:
        str: The filled template as a string.
    """
    template = templates[template_name]
    filled_template = template.format(
        prename=random.choice(first_names),
        surname=random.choice(last_names),
        iban=random.choice(ibans) if "IBAN" in template else "",
        car_number=random.choice(car_numbers) if "CAR NUMBER" in template else "",
        date=random.choice(dates)
    )
    return filled_template

def save_as_pdf(text, filename):
    """
    Saves the given text as a PDF file.

    Args:
        text (str): The text to be saved in the PDF.
        filename (str): The path to the PDF file to be created.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)

def convert_pdf_to_jpg(pdf_filename):
    """
    Converts a PDF file to a JPEG image.

    Args:
        pdf_filename (str): The path to the PDF file to be converted.

    Returns:
        bytes: The binary content of the converted JPEG image.
    """
    images = convert_from_path(pdf_filename)
    jpg_filename = pdf_filename.replace('.pdf', '.jpg')
    for image in images:
        image.save(jpg_filename, 'JPEG')
    with open(jpg_filename, "rb") as image_file:
        return image_file.read()

def generate_single_dataframe(index, num_entries):
    """
    Generates a single dataframe containing randomly generated documents and their corresponding metadata.

    Args:
        index (int): The index of the dataframe.
        num_entries (int): The number of entries (documents) to generate.

    Returns:
        tuple: A tuple containing the index and the generated dataframe.
    """
    series = []
    for _ in range(num_entries):
        class_name = random.choice(list(templates.keys()))
        document_text = generate_document(class_name)
        pdf_filename = os.path.join(output_dir, f"{class_name}_{index}.pdf")
        save_as_pdf(document_text, pdf_filename)
        image_binary = convert_pdf_to_jpg(pdf_filename)
        extracted_text = document_text
        os.remove(pdf_filename)  # Clean up the PDF file

        item = {
            "numerical_id": random.randint(0, 50000),
            "text": extracted_text,
            "page": image_binary,
            "label2": class_name
        }
        series.append(pd.Series(item))
    df = pd.DataFrame(series)
    return (index, df)

def save_dataframe(args, output_dir):
    """
    Saves a dataframe as a Parquet file.

    Args:
        args (tuple): A tuple containing the index and the dataframe.
        output_dir (str): The directory where the Parquet file will be saved.
    """
    index, df = args
    df.to_parquet(os.path.join(output_dir, f"{index}.parquet"))

def generate_data_set(output_dir: str, num_pds: int = 10, num_entries=10):
    """
    Generates a dataset by creating multiple dataframes in parallel.

    Args:
        output_dir (str): The directory where the generated dataframes will be saved.
        num_pds (int): The number of dataframes to generate.
        num_entries (int): The number of entries (documents) per dataframe.
    """
    num_processes = min(num_pds, cpu_count())
    with Pool(processes=num_processes) as pool:
        results = pool.starmap_async(
            generate_single_dataframe, [(i, num_entries) for i in range(num_pds)]
        )
        for result in results.get():
            save_dataframe(result, output_dir)

if __name__ == "__main__":
    """
    Main entry point of the script. Generates the dataset and saves it to the specified directory.
    """
    print("Generating dataset...")
    print(first_names)
    generate_data_set(output_dir, num_pds=10, num_entries=1000)
    print("Data generation completed.")
    os.system("python create_labelencoder.py")