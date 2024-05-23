import streamlit as st
import regression 


def perform_math_operation(param1, param2, integer_input):
    
    result = param1 + param2 * integer_input
    return result

def main():
    st.title('Fipe')
    
    codigoFipe = ""
    modelo = st.text_input('Modelo')
    anoModelo1 = st.number_input('Ano Modelo', step=1)
    anoReferencia1 = st.number_input('Ano Referência', step=1)
    valor1 = st.number_input('Valor')
    
    anoReferencia2 = st.number_input('Ano Previsão', step=1)
    
    
    if modelo and anoModelo1 and anoReferencia1 and valor1 and anoReferencia2:
        
        result = regression.predictFipe(codigoFipe = codigoFipe, 
                  modelo = modelo, 
                  anoModelo1 = anoModelo1,
                  anoReferencia1 = anoReferencia1,
                  valor1 = valor1,
                  anoModelo2 = anoModelo1,
                  anoReferencia2 = anoReferencia2)
        
        st.write('Previsão de Valor:', result)

if __name__ == "__main__":
    main()