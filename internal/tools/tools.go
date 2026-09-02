package tools

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/AlleyBo55/gocode/internal/models"
	"github.com/AlleyBo55/gocode/internal/permissions"
)

// ErrToolNotFound is returned when a tool lookup fails.
var ErrToolNotFound = errors.New("tool not found")

// simpleToolNames defines the tools available in simple mode.
var simpleToolNames = map[string]struct{}{
	"bashtool":     {},
	"filereadtool": {},
	"fileedittool": {},
}

// ToolExecution holds the result of executing a tool.
type ToolExecution struct {
	Name       string `json:"name"`
	SourceHint string `json:"source_hint"`
	Payload    string `json:"payload"`
	Handled    bool   `json:"handled"`
	Message    string `json:"message"`
}

// ToolLookup defines the interface for tool registry operations.
type ToolLookup interface {
	GetTool(name string) (*models.PortingModule, error)
	FindTools(query string, limit int) []models.PortingModule
	GetTools(simpleMode, includeMCP bool, pc permissions.PermissionChecker) []models.PortingModule
	ExecuteTool(name, payload string) ToolExecution
	RenderIndex(limit int, query string) string
	GetToolDefinitions() []models.ToolDefinition
}

// schemaPropertyJSON matches a JSON Schema property definition.
type schemaPropertyJSON struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// inputSchemaJSON matches the input_schema field in tools.json.
type inputSchemaJSON struct {
	Type       string                        `json:"type"`
	Properties map[string]schemaPropertyJSON `json:"properties,omitempty"`
	Required   []string                      `json:"required,omitempty"`
}

// toolJSON matches the JSON shape in tools.json.
type toolJSON struct {
	Name           string          `json:"name"`
	Responsibility string          `json:"responsibility"`
	Description    string          `json:"description"`
	SourceHint     string          `json:"source_hint"`
	InputSchema    inputSchemaJSON `json:"input_schema"`
}

// ToolRegistry implements ToolLookup by loading tools from JSON.
type ToolRegistry struct {
	tools       []models.PortingModule
	index       map[string]*models.PortingModule
	definitions []models.ToolDefinition
}

// NewToolRegistry parses a JSON array of tool objects into a ToolRegistry.
func NewToolRegistry(jsonData []byte) (*ToolRegistry, error) {
	var raw []toolJSON
	if err := json.Unmarshal(jsonData, &raw); err != nil {
		return nil, fmt.Errorf("parsing tools JSON: %w", err)
	}

	toolMods := make([]models.PortingModule, len(raw))
	idx := make(map[string]*models.PortingModule, len(raw))
	defs := make([]models.ToolDefinition, len(raw))

	for i, r := range raw {
		toolMods[i] = models.PortingModule{
			Name:           r.Name,
			Responsibility: r.Responsibility,
			Description:    r.Description,
			SourceHint:     r.SourceHint,
			Status:         "mirrored",
		}
		idx[strings.ToLower(toolMods[i].Name)] = &toolMods[i]

		// Build MCP-compliant ToolDefinition
		props := make(map[string]models.SchemaProperty, len(r.InputSchema.Properties))
		for k, v := range r.InputSchema.Properties {
			props[k] = models.SchemaProperty{
				Type:        v.Type,
				Description: v.Description,
			}
		}
		desc := r.Description
		if desc == "" {
			desc = r.Responsibility
		}
		defs[i] = models.ToolDefinition{
			Name:        r.Name,
			Description: desc,
			InputSchema: models.InputSchema{
				Type:       r.InputSchema.Type,
				Properties: props,
				Required:   r.InputSchema.Required,
			},
		}
	}

	return &ToolRegistry{
		tools:       toolMods,
		index:       idx,
		definitions: defs,
	}, nil
}

// GetToolDefinitions returns MCP-compliant tool definitions for tools/list.
func (tr *ToolRegistry) GetToolDefinitions() []models.ToolDefinition {
	result := make([]models.ToolDefinition, len(tr.definitions))
	copy(result, tr.definitions)
	return result
}

// GetTool returns the tool matching name (case-insensitive), or ErrToolNotFound.
func (tr *ToolRegistry) GetTool(name string) (*models.PortingModule, error) {
	tool, ok := tr.index[strings.ToLower(name)]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrToolNotFound, name)
	}
	return tool, nil
}

// FindTools returns tools whose name or source_hint contain query (case-insensitive),
// limited to at most limit results. If limit <= 0, all matches are returned.
func (tr *ToolRegistry) FindTools(query string, limit int) []models.PortingModule {
	if query == "" {
		if limit > 0 && limit < len(tr.tools) {
			result := make([]models.PortingModule, limit)
			copy(result, tr.tools[:limit])
			return result
		}
		result := make([]models.PortingModule, len(tr.tools))
		copy(result, tr.tools)
		return result
	}

	q := strings.ToLower(query)
	var results []models.PortingModule
	for _, tool := range tr.tools {
		if limit > 0 && len(results) >= limit {
			break
		}
		if strings.Contains(strings.ToLower(tool.Name), q) ||
			strings.Contains(strings.ToLower(tool.SourceHint), q) {
			results = append(results, tool)
		}
	}
	return results
}

// GetTools returns tools filtered by mode flags and permission checker.
func (tr *ToolRegistry) GetTools(simpleMode, includeMCP bool, pc permissions.PermissionChecker) []models.PortingModule {
	var results []models.PortingModule
	for _, tool := range tr.tools {
		nameLower := strings.ToLower(tool.Name)
		hintLower := strings.ToLower(tool.SourceHint)

		if simpleMode {
			if _, ok := simpleToolNames[nameLower]; !ok {
				continue
			}
		}

		if !includeMCP {
			if strings.Contains(nameLower, "mcp") || strings.Contains(hintLower, "mcp") {
				continue
			}
		}

		if pc != nil && pc.IsBlocked(tool.Name) {
			continue
		}

		results = append(results, tool)
	}
	return results
}

// FilterByPermissions returns tools not blocked by the given PermissionChecker.
func (tr *ToolRegistry) FilterByPermissions(pc permissions.PermissionChecker) []models.PortingModule {
	var results []models.PortingModule
	for _, tool := range tr.tools {
		if !pc.IsBlocked(tool.Name) {
			results = append(results, tool)
		}
	}
	return results
}

// ExecuteTool returns a ToolExecution result for the named tool.
// If the tool is not found, Handled is false and Message describes the error.
func (tr *ToolRegistry) ExecuteTool(name, payload string) ToolExecution {
	tool, err := tr.GetTool(name)
	if err != nil {
		return ToolExecution{
			Name:    name,
			Payload: payload,
			Handled: false,
			Message: fmt.Sprintf("tool not found: %s", name),
		}
	}
	return ToolExecution{
		Name:       tool.Name,
		SourceHint: tool.SourceHint,
		Payload:    payload,
		Handled:    true,
		Message:    fmt.Sprintf("executed tool: %s", tool.Name),
	}
}

// RenderIndex returns a Markdown-formatted tool index.
func (tr *ToolRegistry) RenderIndex(limit int, query string) string {
	tools := tr.FindTools(query, limit)

	var sb strings.Builder
	sb.WriteString("# Tool Index\n\n")
	if len(tools) == 0 {
		sb.WriteString("No tools found.\n")
		return sb.String()
	}

	for _, tool := range tools {
		sb.WriteString(fmt.Sprintf("- **%s** — %s (source: %s)\n", tool.Name, tool.Responsibility, tool.SourceHint))
	}
	return sb.String()
}
