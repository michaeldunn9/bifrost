package anthropic

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/bytedance/sonic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// Haiku 5.5 rejects budget_tokens and non-default sampling parameters. Exercise
// the wire body builders used by both native SDK and provider-neutral requests.
func TestHaiku55RequestBodies(t *testing.T) {
	for _, model := range []string{"claude-haiku-5-5", "anthropic.claude-haiku-5-5"} {
		for _, effort := range []string{"low", "medium", "high", "xhigh", "max"} {
			t.Run(model+"/"+effort, func(t *testing.T) {
				ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
				for _, streaming := range []bool{false, true} {
					cfg := AnthropicRequestBuildConfig{Provider: schemas.Anthropic, IsStreaming: streaming}
					chat := &schemas.BifrostChatRequest{
						Provider: schemas.Anthropic, Model: model,
						Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Reply OK")}}},
						Params: &schemas.ChatParameters{
							Reasoning:   &schemas.ChatReasoning{Effort: schemas.Ptr(effort)},
							Temperature: schemas.Ptr(0.2), TopP: schemas.Ptr(0.5), TopK: schemas.Ptr(10),
						},
					}
					body, err := BuildAnthropicChatRequestBody(ctx, chat, cfg)
					if err != nil {
						t.Fatal(err)
					}
					assertHaiku55Body(t, body, effort)
					if got := providerUtils.GetJSONField(body, "max_tokens").Int(); got != 128000 {
						t.Errorf("default max_tokens = %d, want 128000", got)
					}

					// PydanticAI's /v1/messages integration takes this typed roundtrip.
					var native AnthropicMessageRequest
					input := fmt.Sprintf(`{"model":%q,"max_tokens":4096,"thinking":{"type":"adaptive"},"output_config":{"effort":%q},"temperature":0.2,"top_p":0.5,"top_k":10,"messages":[{"role":"user","content":"Reply OK"}]}`, model, effort)
					if err := sonic.Unmarshal([]byte(input), &native); err != nil {
						t.Fatal(err)
					}
					responses := native.ToBifrostResponsesRequest(ctx)
					body, err = BuildAnthropicResponsesRequestBody(ctx, responses, cfg)
					if err != nil {
						t.Fatal(err)
					}
					assertHaiku55Body(t, body, effort)
					if got := providerUtils.GetJSONField(body, "max_tokens").Int(); got != 4096 {
						t.Errorf("explicit max_tokens = %d, want 4096", got)
					}

					// The raw request path must retain native effort too.
					ctx.SetValue(schemas.BifrostContextKeyUseRawRequestBody, true)
					responses.RawRequestBody = []byte(input)
					body, err = BuildAnthropicResponsesRequestBody(ctx, responses, cfg)
					if err != nil {
						t.Fatal(err)
					}
					assertHaiku55Body(t, body, effort)
					ctx.SetValue(schemas.BifrostContextKeyUseRawRequestBody, false)
				}
			})
		}
	}
}

func assertHaiku55Body(t *testing.T, body []byte, effort string) {
	t.Helper()
	if got := providerUtils.GetJSONField(body, "thinking.type").String(); got != "adaptive" {
		t.Errorf("thinking.type = %q, want adaptive", got)
	}
	if got := providerUtils.GetJSONField(body, "output_config.effort").String(); got != effort {
		t.Errorf("effort = %q, want %q", got, effort)
	}
	for _, field := range []string{"thinking.budget_tokens", "temperature", "top_p", "top_k"} {
		if providerUtils.JSONFieldExists(body, field) {
			t.Errorf("unsupported field %s in %s", field, body)
		}
	}
}

func TestHaiku55NativeDefaultEffort(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	var native AnthropicMessageRequest
	if err := sonic.Unmarshal([]byte(`{"model":"claude-haiku-5-5","max_tokens":4096,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"Reply OK"}]}`), &native); err != nil {
		t.Fatal(err)
	}
	body, err := BuildAnthropicResponsesRequestBody(ctx, native.ToBifrostResponsesRequest(ctx), AnthropicRequestBuildConfig{Provider: schemas.Anthropic})
	if err != nil {
		t.Fatal(err)
	}
	assertHaiku55Body(t, body, "medium")
}

func TestHaiku55NativeEffortWithoutThinking(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	var native AnthropicMessageRequest
	if err := sonic.Unmarshal([]byte(`{"model":"claude-haiku-5-5","max_tokens":4096,"output_config":{"effort":"low"},"messages":[{"role":"user","content":"Reply OK"}]}`), &native); err != nil {
		t.Fatal(err)
	}
	body, err := BuildAnthropicResponsesRequestBody(ctx, native.ToBifrostResponsesRequest(ctx), AnthropicRequestBuildConfig{Provider: schemas.Anthropic})
	if err != nil {
		t.Fatal(err)
	}
	assertHaiku55Body(t, body, "low")
}

func TestHaiku55BudgetAndHaiku45Compatibility(t *testing.T) {
	for _, model := range []string{"claude-haiku-5-5", "claude-haiku-4-5"} {
		t.Run(model, func(t *testing.T) {
			ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
			request := &schemas.BifrostChatRequest{
				Provider: schemas.Anthropic, Model: model,
				Input:  []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Reply OK")}}},
				Params: &schemas.ChatParameters{Temperature: schemas.Ptr(0.2), Reasoning: &schemas.ChatReasoning{MaxTokens: schemas.Ptr(2048)}},
			}
			result, err := ToAnthropicChatRequest(ctx, request)
			if err != nil {
				t.Fatal(err)
			}
			if model == "claude-haiku-5-5" {
				if result.Thinking == nil || result.Thinking.Type != "adaptive" || result.Thinking.BudgetTokens != nil {
					t.Errorf("expected adaptive without budget, got %+v", result.Thinking)
				}
				if result.Temperature != nil {
					t.Error("sampling must be omitted")
				}
			} else {
				if result.Thinking == nil || result.Thinking.Type != "enabled" || result.Thinking.BudgetTokens == nil || *result.Thinking.BudgetTokens != 2048 {
					t.Errorf("4.5 budget changed: %+v", result.Thinking)
				}
				if result.Temperature == nil || *result.Temperature != 0.2 {
					t.Error("4.5 sampling changed")
				}
				if result.MaxTokens != 64000 || SupportsEffortParameter(model) {
					t.Error("4.5 capabilities changed")
				}
			}
			request.Params.Reasoning = &schemas.ChatReasoning{Effort: schemas.Ptr("none")}
			result, err = ToAnthropicChatRequest(ctx, request)
			if err != nil {
				t.Fatal(err)
			}
			if result.Thinking == nil || result.Thinking.Type != "disabled" {
				t.Errorf("thinking off changed: %+v", result.Thinking)
			}
		})
	}
}
