import json

def validate_uic(uic_str: str) -> bool:
    """
    Validates a UIC wagon number using the Modulo 10 algorithm.
    The input should contain exactly 12 digits (ignoring spaces/dashes).
    """
    # Remove any non-digit characters
    digits = [int(d) for d in uic_str if d.isdigit()]
    if len(digits) != 12:
        return False
    
    # The first 11 digits are the base number, the 12th is the check digit
    base_digits = digits[:11]
    check_digit = digits[11]
    
    # Modulo 10 algorithm weights: alternating 2 and 1 from right to left (starting at digit 11)
    # Positions 1 to 11 (1-indexed):
    # odd pos: weight 2
    # even pos: weight 1
    weights = [2 if i % 2 == 0 else 1 for i in range(11)]
    
    total = 0
    for i, d in enumerate(base_digits):
        product = d * weights[i]
        # Sum the digits of the product
        total += (product // 10) + (product % 10)
    
    # The calculated check digit is the difference to the next multiple of 10
    calculated_check = (10 - (total % 10)) % 10
    
    return calculated_check == check_digit


def validate_container_id(container_str: str) -> bool:
    """
    Validates a container ID using the ISO 6346 check digit algorithm.
    Input should be 11 alphanumeric characters (e.g. 'CAIU2821420').
    Ignores spaces and dashes.
    """
    # Clean and uppercase
    cleaned = container_str.replace(" ", "").replace("-", "").upper()
    if len(cleaned) != 11:
        return False

    # Must be 4 letters + 6 digits + 1 check digit
    if not (cleaned[:4].isalpha() and cleaned[4:].isdigit()):
        return False

    # Letter-to-number mapping: A=10, B=12, C=13 ... skipping multiples of 11
    def letter_to_num(c):
        val = ord(c) - ord('A') + 10
        val += val // 11  # skip multiples of 11 (11, 22, 33...)
        return val

    # Convert first 10 characters to numeric values
    values = []
    for c in cleaned[:10]:
        values.append(letter_to_num(c) if c.isalpha() else int(c))

    # Each value is multiplied by 2^position (position 0-indexed from left)
    total = sum(v * (2 ** i) for i, v in enumerate(values))

    calculated_check = total % 11 % 10
    return calculated_check == int(cleaned[10])


def form_json(uic_data: dict, wheels_data: list, signs_data: list) -> str:
    """
    Combines the detection results and validates the UIC code.
    Outputs a consolidated JSON string.
    """
    uic_str = uic_data.get("uic_text", "")
    is_valid = validate_uic(uic_str)
    
    output = {
        "status": "success" if is_valid else "warning",
        "message": "UIC validated successfully." if is_valid else "UIC validation failed.",
        "results": {
            "uic": {
                "detected_text": uic_str,
                "confidence": uic_data.get("confidence", 0.0),
                "is_valid": is_valid
            },
            "wheels_count": len(wheels_data),
            "dangerous_goods": [sign.get("label") for sign in signs_data]
        }
    }
    
    return json.dumps(output, indent=4)
