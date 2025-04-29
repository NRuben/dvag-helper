package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	defaultGPTModel    = "o4-mini"
	defaultGeminiModel = "gemini-2.5-pro-exp-03-25"
	defaultProvider    = string(OpenAI)
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
type ProviderEnum string

const (
	OpenAI ProviderEnum = "openai"
	Google ProviderEnum = "google"
)

func ProviderEnumFromString(s string) ProviderEnum {
	switch s {
	case "openai":
		return OpenAI
	case "google":
		return Google
	default:
		fmt.Fprintf(os.Stderr, "[WARN] invalid provider: %s", s)
		fmt.Fprintf(os.Stderr, "[INFO] defaulting to provider: %s", Google)
		return OpenAI
	}
}

// config holds the application configuration
type config struct {
	Model    string
	Mode     Mode
	Provider ProviderEnum
}

// Provider defines the interface for AI providers
type Provider interface {
	GenerateMessage(prompt string) (string, error)
}

// OpenAIProvider implements the Provider interface for OpenAI
type OpenAIProvider struct {
	APIKey string
	APIURL string
	Model  string
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

// --- Google Gemini Provider ---

type GoogleGeminiProvider struct {
	APIKey  string
	BaseURL string
	Model   string
}

type GeminiRequestBody struct {
	Contents []GeminiContent `json:"contents"`
	// GenerationConfig *GeminiGenerationConfig `json:"generationConfig,omitempty"`
}
type GeminiContent struct {
	Parts []GeminiPart `json:"parts"`
	// Role string // Optional: "user" or "model" - defaults to user if omitted in first turn
}
type GeminiPart struct {
	Text string `json:"text"`
}

type GeminiResponseBody struct {
	Candidates     []GeminiCandidate     `json:"candidates"`
	PromptFeedback *GeminiPromptFeedback `json:"promptFeedback,omitempty"` // For potential blocks
}
type GeminiCandidate struct {
	Content       *GeminiContent       `json:"content"`      // Note: Pointer, might be nil
	FinishReason  string               `json:"finishReason"` // e.g., "STOP", "MAX_TOKENS", "SAFETY"
	SafetyRatings []GeminiSafetyRating `json:"safetyRatings"`
}
type GeminiSafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"` // e.g., "NEGLIGIBLE", "LOW", "MEDIUM", "HIGH"
}
type GeminiPromptFeedback struct {
	BlockReason   string               `json:"blockReason,omitempty"` // If the prompt was blocked
	SafetyRatings []GeminiSafetyRating `json:"safetyRatings"`
}

func (p *GoogleGeminiProvider) GenerateMessage(prompt string) (string, error) {
	apiURL := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s", p.BaseURL, p.Model, p.APIKey)

	requestBody := GeminiRequestBody{
		Contents: []GeminiContent{
			{
				Parts: []GeminiPart{
					{
						Text: prompt,
					},
				},
			},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal gemini request: %w", err)
	}

	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create gemini request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to make gemini API request: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read gemini response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var errorResponse struct {
			Error struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
				Status  string `json:"status"`
			} `json:"error"`
		}
		if json.Unmarshal(bodyBytes, &errorResponse) == nil && errorResponse.Error.Message != "" {
			return "", fmt.Errorf("gemini API error (%d %s): %s", errorResponse.Error.Code, errorResponse.Error.Status, errorResponse.Error.Message)
		}
		// Fallback generic error
		return "", fmt.Errorf("gemini API request failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var response GeminiResponseBody
	if err = json.Unmarshal(bodyBytes, &response); err != nil {
		return "", fmt.Errorf("failed to decode gemini response: %w\nResponse body: %s", err, string(bodyBytes))
	}

	// Check if the prompt itself was blocked
	if response.PromptFeedback != nil && response.PromptFeedback.BlockReason != "" {
		return "", fmt.Errorf("gemini prompt blocked due to %s", response.PromptFeedback.BlockReason)
	}

	// Check if response is empty or blocked
	if len(response.Candidates) == 0 {
		// Check finish reason if available (though candidates might be empty before finishReason is set)
		return "", fmt.Errorf("no candidates in gemini API response. Body: %s", string(bodyBytes))
	}

	candidate := response.Candidates[0]
	if candidate.FinishReason != "STOP" && candidate.FinishReason != "MAX_TOKENS" {
		// Other reasons include SAFETY, RECITATION, OTHER
		return "", fmt.Errorf("gemini generation finished due to %s", candidate.FinishReason)
	}

	if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
		return "", fmt.Errorf("gemini response candidate has no content parts. FinishReason: %s", candidate.FinishReason)
	}

	// Concatenate parts if needed, though usually there's one for simple text generation
	var generatedText strings.Builder
	for _, part := range candidate.Content.Parts {
		generatedText.WriteString(part.Text)
	}

	return generatedText.String(), nil
}
func (p *OpenAIProvider) GenerateMessage(prompt string) (string, error) {
	requestBody := OpenAIRequestBody{
		Model: p.Model,
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
func NewProvider(cfg *config) Provider {
	switch cfg.Provider {
	case OpenAI:
		key, ok := os.LookupEnv("OPENAI_API_KEY")

		if !ok || key == "" {
			log.Fatal("API key not set. Please set the OPENAI_API_KEY environment variable.")
		}

		model := defaultGPTModel
		if cfg.Model != "" {
			model = cfg.Model
		}
		return &OpenAIProvider{
			APIKey: key,
			APIURL: "https://api.openai.com/v1/chat/completions",
			Model:  model,
		}
	case Google:
		key, ok := os.LookupEnv("GEMINI_API_KEY")
		if !ok || key == "" {
			log.Fatal("API key not set. Please set the GEMINI_API_KEY environment variable.")
		}
		model := defaultGeminiModel
		if cfg.Model != "" && cfg.Model != defaultGPTModel {
			model = cfg.Model
		}
		return &GoogleGeminiProvider{
			APIKey:  key,
			BaseURL: "https://generativelanguage.googleapis.com",
			Model:   model,
		}
	default:
		log.Fatalf("unsupported provider: %s", cfg.Provider)
		return nil
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
	provider := NewProvider(cfg)

	// Use the provider to generate the message
	return provider.GenerateMessage(prompt)
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
	fmt.Printf("  %s                 OpenAI API (default)\n", OpenAI)
	fmt.Printf("  %s                 Google Gemini\n", Google)
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
	fmt.Printf("  OPENAI_API_KEY: <%s> (required for the provider: %s)\n", keySet, OpenAI)

	keySet = "unset"
	if os.Getenv("GEMINI_API_KEY") != "" {
		keySet = "set"
	}
	fmt.Printf("  GEMINI_API_KEY: <%s> (required for the provider: %s)\n", keySet, Google)
}

func parseFlags() *config {
	cfg := &config{
		Model:    defaultGPTModel,
		Mode:     ModeCommit,
		Provider: ProviderEnumFromString(defaultProvider),
	}

	helpFlag := flag.Bool("help", false, "Display usage information")
	modelFlag := flag.String("model", defaultGPTModel, "AI model to use for generating messages")

	var providerFlag string
	flag.StringVar(&providerFlag, "provider", defaultProvider, "AI provider to use (openai, google etc.)")
	flag.StringVar(&providerFlag, "p", defaultProvider, "AI provider to use (openai, google, etc.)")

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
	cfg.Provider = ProviderEnumFromString(providerFlag)

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
