package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	defaultGPTModel = "o4-mini"
	defaultProvider = "openai"
)

// Prompt templates
const (
	commitPromptTemplate = `Do not use ` + "```" + `.
 Create a CONVENTIONAL commit message for this git diff with the structure: <type>[optional scope]: <description>
Ignore formatting and whitespace changes and focus on the big picture.
%s`

	prPromptTemplate = `Do not use ` + "```" + `.
Create a pull request description for these changes.
Include: 1) A clear title, 2) What changes were made, 3) Why these changes were necessary,
and 4) Any testing considerations.
Ignore formatting and whitespace changes and focus on the big picture.
Write the description in GERMAN!!!
Format with markdown:
%s`
)

type Mode string

const (
	ModeCommit Mode = "commit"
	ModePR     Mode = "pr"
)

func (m Mode) String() string {
	return string(m)
}

// Message represents a message in a conversation with an AI model
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// config holds the application configuration
type config struct {
	Model    string
	Mode     Mode
	Provider string
}

// Provider defines the interface for AI providers
type Provider interface {
	GenerateMessage(prompt string, model string) (string, error)
}

// OpenAIProvider implements the Provider interface for OpenAI
type OpenAIProvider struct {
	APIKey string
	APIURL string
}

type OpenAIRequestBody struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type OpenAIResponseBody struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func (p *OpenAIProvider) GenerateMessage(prompt string, model string) (string, error) {
	requestBody := OpenAIRequestBody{
		Model: model,
		Messages: []Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", p.APIURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to make API request: %w", err)
	}
	defer resp.Body.Close()

	var response OpenAIResponseBody
	if err = json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no choices in API response")
	}

	return response.Choices[0].Message.Content, nil
}

// NewProvider creates and returns a provider based on the configuration
func NewProvider(cfg *config) (Provider, error) {
	switch cfg.Provider {
	case "openai":
		key, ok := os.LookupEnv("OPENAI_API_KEY")

		if !ok || key == "" {
			log.Fatal("API key not set. Please set the OPENAI_API_KEY environment variable.")
		}

		return &OpenAIProvider{
			APIKey: key,
			APIURL: "https://api.openai.com/v1/chat/completions",
		}, nil
	// Add more providers here in the future
	default:
		return nil, fmt.Errorf("unsupported provider: %s", cfg.Provider)
	}
}

func getGitDiff() (string, error) {
	stdinStat, _ := os.Stdin.Stat()
	if (stdinStat.Mode() & os.ModeCharDevice) == 0 {
		stdinBytes, err := os.ReadFile(os.Stdin.Name())
		if err != nil {
			return "", fmt.Errorf("failed to read from stdin: %w", err)
		}

		diff := strings.TrimSpace(string(stdinBytes))
		if diff != "" {
			return diff, nil
		}
	}
	fmt.Fprintf(os.Stderr, "[WARN] No input from stdin, checking for staged changes...\n")
	cmd := exec.Command("git", "diff", "--staged")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to get git diff: %w", err)
	}

	diff := strings.TrimSpace(out.String())
	if diff == "" {
		return "", fmt.Errorf("no staged changes found")
	}
	return diff, nil
}

func generateMessage(cfg *config, gitDiff string) (string, error) {
	var prompt string

	switch cfg.Mode {
	case ModeCommit:
		prompt = fmt.Sprintf(commitPromptTemplate, gitDiff)
	case ModePR:
		prompt = fmt.Sprintf(prPromptTemplate, gitDiff)
	default:
		return "", fmt.Errorf("invalid mode: %s", cfg.Mode)
	}

	// Create a provider based on the configuration
	provider, err := NewProvider(cfg)
	if err != nil {
		return "", fmt.Errorf("failed to create provider: %w", err)
	}

	// Use the provider to generate the message
	return provider.GenerateMessage(prompt, cfg.Model)
}

// printUsage displays detailed help information about the application
func printUsage() {
	appName := filepath.Base(os.Args[0])

	// Header
	fmt.Println("AI-Powered Git Commit and PR Message Generator")
	fmt.Println(strings.Repeat("=", 60))

	// Description
	fmt.Println("\nDESCRIPTION:")
	fmt.Println("  This tool generates conventional commit messages or PR descriptions using AI providers.")
	fmt.Println("  It can read diffs from stdin or use staged git changes.")

	// Usage
	fmt.Println("\nUSAGE:")
	fmt.Printf("  %s [options]\n", appName)

	// Modes
	fmt.Println("\nMODES:")
	fmt.Println("  --cm                   Generate a commit message (default mode)")
	fmt.Println("  --pr                   Generate a pull request description")

	// Options
	fmt.Println("\nOPTIONS:")
	flag.PrintDefaults()

	// Providers
	fmt.Println("\nSUPPORTED PROVIDERS:")
	fmt.Println("  openai                 OpenAI API (default)")
	// Add more providers here as they are implemented

	// Examples
	fmt.Println("\nEXAMPLES:")
	fmt.Printf("  %s                      # Generate commit message from staged changes\n", appName)
	fmt.Printf("  %s --pr                 # Generate PR description from staged changes\n", appName)
	fmt.Printf("  git diff | %s           # Generate commit message from piped git diff\n", appName)
	fmt.Printf("  git diff | %s --pr      # Generate PR description from piped git diff\n", appName)
	fmt.Printf("  %s --model=\"o4-mini\"    # Use a specific AI model\n", appName)
	fmt.Printf("  %s --provider=\"openai\"  # Use a specific AI provider\n", appName)
	fmt.Printf("  %s --help               # Show this help message\n", appName)

	fmt.Println("\nENVIRONMENT VARIABLES:")
	keySet := "unset"
	if os.Getenv("OPENAI_API_KEY") != "" {
		keySet = "set"
	}
	fmt.Printf("  OPENAI_API_KEY: <%s> (required for OpenAI provider)\n", keySet)
	// Add more environment variables here as more providers are implemented
}

func parseFlags() *config {
	cfg := &config{
		Model:    defaultGPTModel,
		Mode:     ModeCommit,
		Provider: defaultProvider,
	}

	helpFlag := flag.Bool("help", false, "Display usage information")
	modelFlag := flag.String("model", defaultGPTModel, "AI model to use for generating messages")
	providerFlag := flag.String("provider", defaultProvider, "AI provider to use (openai, etc.)")

	commitFlag := flag.Bool("cm", false, "Generate a commit message (default mode)")
	prFlag := flag.Bool("pr", false, "Generate a pull request description")

	flag.Parse()

	if *helpFlag {
		printUsage()
		os.Exit(0)
	}

	if *prFlag && *commitFlag {
		fmt.Fprintln(os.Stderr, "[WARN] Both --pr and --cm flags specified. Using --cm mode.")
		cfg.Mode = ModeCommit
	} else if *prFlag {
		cfg.Mode = ModePR
	} else {
		cfg.Mode = ModeCommit
	}

	cfg.Model = *modelFlag
	cfg.Provider = *providerFlag

	return cfg
}

func main() {
	cfg := parseFlags()

	gitDiff, err := getGitDiff()
	if err != nil {
		log.Fatal("Error getting git diff:", err)
	}

	generatedMessage, err := generateMessage(cfg, gitDiff)
	if err != nil {
		log.Fatal("Error generating message:", err)
	}

	fmt.Println(generatedMessage)

}
