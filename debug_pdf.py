import fitz

pdf_path = r"data\Bitcoin 199 domande (Italian Edition) (Alessio Barnini  Alessandro Aglietti) (Z-Library).pdf"
pdf_doc = fitz.open(pdf_path)

# Look for pages with actual content about bitcoin definition
for page_num in range(len(pdf_doc)):
    text = pdf_doc[page_num].get_text()
    if "Che cosa è bitcoin" in text or "cos'è bitcoin" in text.lower() or "definizione" in text.lower():
        print(f"\n{'='*80}")
        print(f"PAGE {page_num + 1}:")
        print('='*80)
        print(text[:800])
        print("\n...")
