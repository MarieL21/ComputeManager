# Creates a source file with string containing input file
FILE(READ "${INPUT_FILE}" DATA_STRING)

STRING(REGEX REPLACE "\\\\" "\\\\\\\\" DATA_STRING "${DATA_STRING}")
STRING(REGEX REPLACE "\"" "\\\\\"" DATA_STRING "${DATA_STRING}")
STRING(REGEX REPLACE "\n" "\\\\n\"\\n\"" DATA_STRING "${DATA_STRING}")

FILE(WRITE ${OUTPUT_FILE}
	"const ${TYPE_MODIFIERS} char *${VARIABLE_NAME} =\n"
	"\"${DATA_STRING}\"; \n")

