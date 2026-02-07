"""
Quick verification script to test PyArrow fix
Run this after restarting your Streamlit app
"""

import pandas as pd
import numpy as np

def test_mixed_type_dataframe():
    """Test that our fix handles mixed types correctly"""
    
    print("=" * 60)
    print("Testing Mixed Type DataFrame (like the error case)")
    print("=" * 60)
    
    # Simulate the problematic data structure
    test_df = pd.DataFrame({
        'Field': ['Amount', 'Gender', 'Age', 'Count'],
        'Value': [
            12345.67,  # numeric
            'F',        # string (this caused the error!)
            45,         # numeric
            'N/A'       # string
        ]
    })
    
    print("\n1. Original DataFrame (mixed types):")
    print(test_df)
    print(f"\nData types:\n{test_df.dtypes}")
    print(f"\nValue column type: {test_df['Value'].dtype}")
    
    # Apply the fix
    test_df_fixed = test_df.copy()
    test_df_fixed['Value'] = test_df_fixed['Value'].astype(str)
    
    print("\n2. Fixed DataFrame (all strings):")
    print(test_df_fixed)
    print(f"\nData types:\n{test_df_fixed.dtypes}")
    print(f"\nValue column type: {test_df_fixed['Value'].dtype}")
    
    print("\n✅ Fix successfully converts all values to strings!")
    print("This prevents PyArrow from trying to infer mixed types.")
    
    # Test PyArrow conversion
    print("\n3. Testing PyArrow conversion:")
    try:
        import pyarrow as pa
        
        print("   - Converting original (should fail)...")
        try:
            table = pa.Table.from_pandas(test_df)
            print("   ⚠️  Unexpected: Original converted without error")
        except Exception as e:
            print(f"   ✅ Expected error occurred: {type(e).__name__}")
            print(f"      Message: {str(e)[:100]}...")
        
        print("\n   - Converting fixed version...")
        try:
            table = pa.Table.from_pandas(test_df_fixed)
            print("   ✅ Fixed version converts successfully!")
            print(f"      Schema: {table.schema}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            
    except ImportError:
        print("   ⚠️  PyArrow not installed, skipping direct conversion test")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


def simulate_streamlit_scenario():
    """Simulate the exact scenario from the webapp"""
    
    print("\n\n" + "=" * 60)
    print("Simulating Streamlit Webapp Scenario")
    print("=" * 60)
    
    # Simulate claim data
    claim_data = {
        'claim_amount': 5432.10,
        'claim_type': 'Medical',
        'claim_duration_days': 15,
        'geographic_region': 'Northeast',
        'diagnosis_code': 'D1234',
        'procedure_code': 'P5678',
        'patient_age': 45,
        'patient_gender': 'F',  # This is the culprit!
        'num_previous_claims': 2,
        'average_claim_amount': 3200.50,
        'days_since_last_claim': 90
    }
    
    print("\n1. Creating info_df (Claim Information):")
    info_df = pd.DataFrame({
        'Field': ['Amount', 'Type', 'Duration', 'Region', 'Diagnosis', 'Procedure'],
        'Value': [
            f"${claim_data.get('claim_amount', 'N/A'):,.2f}",
            claim_data.get('claim_type', 'N/A'),
            f"{claim_data.get('claim_duration_days', 'N/A')} days",
            claim_data.get('geographic_region', 'N/A'),
            claim_data.get('diagnosis_code', 'N/A'),
            claim_data.get('procedure_code', 'N/A')
        ]
    })
    print(info_df)
    print(f"Value dtype: {info_df['Value'].dtype}")
    
    print("\n2. Creating patient_df (Patient Information) - PROBLEMATIC:")
    patient_df = pd.DataFrame({
        'Field': ['Age', 'Gender', 'Previous Claims', 'Avg Claim Amount', 'Days Since Last'],
        'Value': [
            claim_data.get('patient_age', 'N/A'),
            claim_data.get('patient_gender', 'N/A'),  # 'F' - the string that caused issues!
            claim_data.get('num_previous_claims', 'N/A'),
            f"${claim_data.get('average_claim_amount', 'N/A'):,.2f}",
            f"{claim_data.get('days_since_last_claim', 'N/A')} days"
        ]
    })
    print(patient_df)
    print(f"Value dtype: {patient_df['Value'].dtype}")
    print("⚠️  Note: 'Value' column has dtype 'object' (mixed types)")
    print("    Gender='F' (string) mixed with Age=45 (int)")
    
    print("\n3. Applying the FIX:")
    patient_df['Value'] = patient_df['Value'].astype(str)
    print(patient_df)
    print(f"Value dtype: {patient_df['Value'].dtype}")
    print("✅ Now all values are consistently strings!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_mixed_type_dataframe()
    simulate_streamlit_scenario()
    
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The fix works by:
1. Creating DataFrames with mixed types (numeric + string)
2. Explicitly converting the 'Value' column to string type
3. This prevents PyArrow from trying to infer a single type
4. Streamlit can now successfully serialize the table

To verify in your app:
1. Restart Streamlit
2. Go to Individual Analysis tab
3. Tables should now display without errors!
""")
