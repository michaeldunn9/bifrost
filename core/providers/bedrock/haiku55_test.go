package bedrock

import (
	"context"
	"testing"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

func TestHaiku55ConverseConversion(t *testing.T) {
	for _, model := range []string{"anthropic.claude-haiku-5-5", "global.anthropic.claude-haiku-4-5-20251001-v1:0"} {
		t.Run(model, func(t *testing.T) {
			ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
			chat, err := ToBedrockChatCompletionRequest(ctx, &schemas.BifrostChatRequest{
				Provider: schemas.Bedrock, Model: model,
				Input:  []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Reply OK")}}},
				Params: &schemas.ChatParameters{Reasoning: &schemas.ChatReasoning{Effort: schemas.Ptr("low")}, Temperature: schemas.Ptr(0.2), TopP: schemas.Ptr(0.5)},
			})
			if err != nil {
				t.Fatal(err)
			}
			responses, err := ToBedrockResponsesRequest(ctx, &schemas.BifrostResponsesRequest{
				Provider: schemas.Bedrock, Model: model,
				Input:  []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("Reply OK")}}},
				Params: &schemas.ResponsesParameters{Reasoning: &schemas.ResponsesParametersReasoning{Effort: schemas.Ptr("low")}, Temperature: schemas.Ptr(0.2), TopP: schemas.Ptr(0.5)},
			})
			if err != nil {
				t.Fatal(err)
			}
			for _, request := range []*BedrockConverseRequest{chat, responses} {
				body, err := providerUtils.MarshalSorted(request)
				if err != nil {
					t.Fatal(err)
				}
				if model == "anthropic.claude-haiku-5-5" {
					if got := providerUtils.GetJSONField(body, "additionalModelRequestFields.thinking.type").String(); got != "adaptive" {
						t.Errorf("thinking.type = %q", got)
					}
					if got := providerUtils.GetJSONField(body, "additionalModelRequestFields.output_config.effort").String(); got != "low" {
						t.Errorf("effort = %q", got)
					}
					for _, field := range []string{"inferenceConfig.temperature", "inferenceConfig.topP", "additionalModelRequestFields.thinking.budget_tokens"} {
						if providerUtils.JSONFieldExists(body, field) {
							t.Errorf("unexpected %s", field)
						}
					}
				} else {
					if got := providerUtils.GetJSONField(body, "additionalModelRequestFields.thinking.type").String(); got != "enabled" {
						t.Errorf("4.5 thinking changed: %q", got)
					}
					if !providerUtils.JSONFieldExists(body, "inferenceConfig.temperature") {
						t.Error("4.5 sampling changed")
					}
				}
			}
		})
	}
}
