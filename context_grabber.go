package main

import (
	"bufio"
	"compress/gzip"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// --- Configuration ---

const (
	scriptFileName  = "context_grabber.go" // Name of this script file
	VENV_DIR        = ".venv"              // Directory to ignore
	GIT_DIR         = ".git"               // Directory to ignore
	PYCACHE_DIR     = "__pycache__"        // Python cache dir to ignore
	DATA_DIR        = "data"               // Used only implicitly for patterns
	PY_EXT          = ".py"
	CSV_EXT         = ".csv"
	CSV_GZ_EXT      = ".csv.gz"
	PARQUET_EXT     = ".parquet"
	YAML_EXT1       = ".yaml"
	YAML_EXT2       = ".yml"
	ENV_EXT         = ".env"
	SAFETENSORS_EXT = ".safetensors"
	PT_EXT          = ".pt"
	JSON_EXT        = ".json"
	CU_EXT          = ".cu"
	CUH_EXT         = ".cuh"
	NPZ_EXT         = ".npz"
	outputFilename  = "context_output.txt"
	rootDir         = "."
)

// --- Helpers ---

// isDateSubdir checks if path is a date-based subdir under bars_* or btc_parquet_clean.
func isDateSubdir(path string) bool {
	base := filepath.Base(path)
	if !strings.HasPrefix(base, "d=") {
		return false
	}
	parent := filepath.Base(filepath.Dir(path))
	return strings.HasPrefix(parent, "bars_") || parent == "btc_parquet_clean"
}

// getLanguageIdentifier returns a language tag for fenced code blocks based on filename.
func getLanguageIdentifier(filename string) string {
	lower := strings.ToLower(filename)
	var ext string
	if strings.HasSuffix(lower, ".gz") {
		inner := strings.TrimSuffix(lower, ".gz")
		ext = filepath.Ext(inner)
	} else {
		ext = filepath.Ext(lower)
	}

	switch ext {
	case ".py":
		return "python"
	case ".yaml", ".yml":
		return "yaml"
	case ".env":
		return "bash"
	case ".csv":
		return "csv"
	case ".go":
		return "go"
	case ".js":
		return "javascript"
	case ".ts":
		return "typescript"
	case ".java":
		return "java"
	case ".cpp", ".cc", ".cxx":
		return "cpp"
	case ".c":
		return "c"
	case ".cs":
		return "csharp"
	case ".rb":
		return "ruby"
	case ".php":
		return "php"
	case ".swift":
		return "swift"
	case ".rs":
		return "rust"
	case ".sh":
		return "bash"
	case ".sql":
		return "sql"
	case ".json":
		return "json"
	case ".xml":
		return "xml"
	case ".html":
		return "html"
	case ".css":
		return "css"
	case ".md":
		return "markdown"
	case ".cu", ".cuh":
		return "cpp"
	case ".npz":
		return "text"
	default:
		return ""
	}
}

// getFileContent reads full content or first n lines from path.
// If full==false, reads up to n lines using a Scanner (memory-friendly).
// Handles .gz transparently.
func getFileContent(path string, full bool, n int) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("opening file %s: %w", path, err)
	}
	defer f.Close()

	lowerPath := strings.ToLower(path)

	var r io.Reader = f
	if strings.HasSuffix(lowerPath, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return "", fmt.Errorf("creating gzip reader for %s: %w", path, err)
		}
		defer gz.Close()
		r = gz
	}

	if full {
		b, err := io.ReadAll(r)
		if err != nil {
			return "", fmt.Errorf("reading full file %s: %w", path, err)
		}
		return string(b), nil
	}

	var sb strings.Builder
	scanner := bufio.NewScanner(r)
	count := 0
	for count < n && scanner.Scan() {
		sb.WriteString(scanner.Text())
		sb.WriteString("\n")
		count++
	}
	if err := scanner.Err(); err != nil {
		return sb.String(), fmt.Errorf("scanning file %s: %w", path, err)
	}

	if count < n && count > 0 {
		sb.WriteString(fmt.Sprintf("# (File has only %d lines)\n", count))
	} else if count == 0 {
		sb.WriteString("# (File is empty or could not be read)\n")
	}

	return sb.String(), nil
}

// isRelevantFile determines which files are listed and considered for content dumping.
func isRelevantFile(name string) bool {
	lower := strings.ToLower(name)

	// FIX: Explicitly ignore temp files here too if passed directly
	if strings.HasSuffix(lower, "~") || name == outputFilename {
		return false
	}

	if strings.HasSuffix(lower, CSV_GZ_EXT) {
		return true
	}

	switch filepath.Ext(lower) {
	case PY_EXT,
		".go",
		CSV_EXT,
		PARQUET_EXT,
		YAML_EXT1,
		YAML_EXT2,
		ENV_EXT,
		SAFETENSORS_EXT,
		PT_EXT,
		JSON_EXT,
		CU_EXT,
		CUH_EXT,
		NPZ_EXT:
		return true
	default:
		return false
	}
}

func main() {
	log.SetOutput(os.Stderr)
	log.SetFlags(log.LstdFlags)

	// OPTIMIZATION: Open output file immediately to stream content
	// instead of holding everything in a massive string builder.
	outFile, err := os.Create(outputFilename)
	if err != nil {
		log.Fatalf("Failed to create %s: %v", outputFilename, err)
	}
	defer outFile.Close()

	writer := bufio.NewWriter(outFile)
	defer writer.Flush()

	var relevantFilePaths []string

	log.Println("Starting: building file tree and collecting relevant paths...")

	// Write tree header immediately
	writer.WriteString("--- File Tree Structure ---\n")

	err = filepath.WalkDir(rootDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			log.Printf("Error accessing %s: %v (skipping)", path, err)
			return nil
		}

		name := d.Name()

		// FIX: Skip the script file AND the output file AND temp files
		if name == scriptFileName || name == outputFilename {
			return nil
		}
		if strings.HasSuffix(name, "~") || strings.HasSuffix(name, ".swp") || name == ".DS_Store" {
			return nil
		}

		// Skip certain directories entirely
		if d.IsDir() {
			switch name {
			case VENV_DIR, GIT_DIR, PYCACHE_DIR:
				log.Printf("Skipping directory: %s", path)
				return filepath.SkipDir
			}
			if isDateSubdir(path) {
				log.Printf("Skipping date subdir: %s", path)
				return filepath.SkipDir
			}
		}

		// Skip root itself in output
		if path == rootDir {
			return nil
		}

		rel, relErr := filepath.Rel(rootDir, path)
		if relErr != nil {
			log.Printf("Rel path error for %s: %v, using absolute", path, relErr)
			rel = path
		}

		isDir := d.IsDir()
		isFileRelevant := !isDir && isRelevantFile(name)

		if isDir || isFileRelevant {
			depth := strings.Count(rel, string(filepath.Separator))
			indent := strings.Repeat("    ", depth) + "|-- "
			displayName := name
			if isDir {
				displayName += "/"
			}

			// Write directly to file buffer
			writer.WriteString(indent)
			writer.WriteString(displayName)
			writer.WriteString("\n")

			if isFileRelevant && !isDir {
				relevantFilePaths = append(relevantFilePaths, path)
			}
		}

		return nil
	})

	// Add a separating newline after the tree
	writer.WriteString("\n")

	if err != nil {
		log.Fatalf("Error while walking directory tree: %v", err)
	}

	// Flush after tree generation
	writer.Flush()

	log.Printf("Tree build complete. Relevant files: %d", len(relevantFilePaths))

	log.Println("Processing file contents for context...")

	for _, path := range relevantFilePaths {
		rel, err := filepath.Rel(rootDir, path)
		if err != nil {
			log.Printf("Rel path error for %s: %v, using absolute", path, err)
			rel = path
		}

		lowerPath := strings.ToLower(path)

		// Skip content for large/binary-ish data formats; tree listing is enough.
		if strings.HasSuffix(lowerPath, CSV_EXT) ||
			strings.HasSuffix(lowerPath, CSV_GZ_EXT) ||
			strings.HasSuffix(lowerPath, PARQUET_EXT) ||
			strings.HasSuffix(lowerPath, NPZ_EXT) {
			continue
		}

		// Decide if we read full content.
		shouldReadFull := strings.HasSuffix(lowerPath, PY_EXT) ||
			strings.HasSuffix(lowerPath, ".go") ||
			strings.HasSuffix(lowerPath, YAML_EXT1) ||
			strings.HasSuffix(lowerPath, YAML_EXT2) ||
			strings.HasSuffix(lowerPath, ENV_EXT) ||
			strings.HasSuffix(lowerPath, CU_EXT) ||
			strings.HasSuffix(lowerPath, CUH_EXT) ||
			strings.HasSuffix(lowerPath, SAFETENSORS_EXT) ||
			strings.HasSuffix(lowerPath, PT_EXT)

		if !shouldReadFull {
			continue
		}

		langID := getLanguageIdentifier(path)

		log.Printf("Including file: %s", rel)
		writer.WriteString("// --- File: ")
		writer.WriteString(rel)
		writer.WriteString(" ---\n\n")

		content, err := getFileContent(path, true, 0)
		if err != nil {
			log.Printf("Error reading %s: %v", path, err)
			writer.WriteString("// Error reading file: ")
			writer.WriteString(err.Error())
			writer.WriteString("\n\n")
		} else {
			writer.WriteString("```")
			writer.WriteString(langID)
			writer.WriteString("\n")
			writer.WriteString(content)
			if len(content) > 0 && content[len(content)-1] != '\n' {
				writer.WriteString("\n")
			}
			writer.WriteString("```\n\n")
		}

		writer.WriteString("// --- End File: ")
		writer.WriteString(rel)
		writer.WriteString(" ---\n\n")

		// Flush periodically to keep memory usage stable
		writer.Flush()
	}

	log.Println("File content collection complete.")
	log.Printf("Successfully wrote context to %s", outputFilename)
}
