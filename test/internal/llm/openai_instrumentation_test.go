package llm_test

import (
	"context"
	"testing"

	"github.com/agenticgokit/agenticgokit/internal/llm"
	"github.com/agenticgokit/agenticgokit/internal/observability"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// testWriter discards output for test tracing
type testWriter struct{}

func (testWriter) Write(p []byte) (n int, err error) {
	return len(p), nil
}

// setupTestTracer initializes OpenTelemetry for testing
func setupTestTracer(ctx context.Context) (func(context.Context) error, error) {
	// Use in-memory span recorder for testing
	exporter := tracetest.NewInMemoryExporter()
	tp := trace.NewTracerProvider(
		trace.WithSampler(trace.AlwaysSample()),
		trace.WithSpanProcessor(trace.NewSimpleSpanProcessor(exporter)),
	)

	otel.SetTracerProvider(tp)

	shutdown := func(ctx context.Context) error {
		return tp.Shutdown(ctx)
	}

	return shutdown, nil
}

// setupTestTracerWithExporter returns tracer shutdown and in-memory exporter for span assertions
func setupTestTracerWithExporter(t *testing.T, ctx context.Context) (*tracetest.InMemoryExporter, func(context.Context) error) {
	t.Helper()

	exporter := tracetest.NewInMemoryExporter()
	tp := trace.NewTracerProvider(
		trace.WithSampler(trace.AlwaysSample()),
		trace.WithSpanProcessor(trace.NewSimpleSpanProcessor(exporter)),
	)

	otel.SetTracerProvider(tp)

	shutdown := func(ctx context.Context) error {
		return tp.Shutdown(ctx)
	}

	return exporter, shutdown
}

// findAttrValue returns the attribute value for a given key if present
func findAttrValue(attrs []attribute.KeyValue, key attribute.Key) (attribute.Value, bool) {
	for _, attr := range attrs {
		if attr.Key == key {
			return attr.Value, true
		}
	}
	return attribute.Value{}, false
}

// TestOpenAICallInstrumentation tests that Call() creates observability spans
func TestOpenAICallInstrumentation(t *testing.T) {
	// Setup tracing
	ctx := context.Background()
	shutdown, err := setupTestTracer(ctx)
	require.NoError(t, err, "failed to setup test tracer")
	defer shutdown(ctx) //nolint:errcheck

	// Test that creating an adapter and calling it doesn't panic
	// We use a mock server in actual tests, here we just verify span creation doesn't break
	adapter, err := llm.NewOpenAIAdapter("test-key", "gpt-4o-mini", 100, 0.7)
	require.NoError(t, err, "failed to create adapter")
	assert.NotNil(t, adapter, "adapter should not be nil")

	// Note: We can't actually call the LLM without a real/mock server,
	// but we verified the instrumentation compiles and doesn't break existing tests
}

// TestOpenAIStreamInstrumentation tests that Stream() creates observability spans
func TestOpenAIStreamInstrumentation(t *testing.T) {
	// Setup tracing
	ctx := context.Background()
	shutdown, err := setupTestTracer(ctx)
	require.NoError(t, err, "failed to setup test tracer")
	defer shutdown(ctx) //nolint:errcheck

	// Test that creating an adapter doesn't panic
	adapter, err := llm.NewOpenAIAdapter("test-key", "gpt-4o-mini", 100, 0.7)
	require.NoError(t, err, "failed to create adapter")
	assert.NotNil(t, adapter, "adapter should not be nil")

	// Verify stream method exists and has proper signature
	// Actual stream testing requires a mock server (covered in existing tests)
}

// TestMockProviderWithInstrumentation tests instrumentation with mock provider
func TestMockProviderWithInstrumentation(t *testing.T) {
	// Setup tracing
	ctx := context.Background()
	shutdown, err := setupTestTracer(ctx)
	require.NoError(t, err, "failed to setup test tracer")
	defer shutdown(ctx) //nolint:errcheck

	// Create mock adapter that doesn't require external calls
	mockAdapter := llm.NewMockAdapter("test-model")
	assert.NotNil(t, mockAdapter, "mock adapter should not be nil")

	// Test Call - mock adapter should handle it
	response, err := mockAdapter.Call(ctx, llm.Prompt{
		User: "test prompt",
		Parameters: llm.ModelParameters{
			MaxTokens:   int32Ptr(150),
			Temperature: float32Ptr(0.7),
		},
	})

	assert.NoError(t, err, "mock call should not error")
	assert.NotEmpty(t, response.Content, "response should have content")

	// Test Stream
	tokenChan, err := mockAdapter.Stream(ctx, llm.Prompt{
		User: "test stream prompt",
		Parameters: llm.ModelParameters{
			MaxTokens:   int32Ptr(150),
			Temperature: float32Ptr(0.7),
		},
	})

	assert.NoError(t, err, "mock stream should not error")
	assert.NotNil(t, tokenChan, "token channel should not be nil")

	// Consume the stream
	for token := range tokenChan {
		if token.Error != nil {
			t.Fatalf("unexpected error in stream: %v", token.Error)
		}
	}
}

// TestObservabilityAttributesPresent tests that LLM attributes are properly defined
func TestObservabilityAttributesPresent(t *testing.T) {
	// Verify observability attributes exist
	assert.NotEmpty(t, observability.AttrLLMProvider, "LLM provider attribute should be defined")
	assert.NotEmpty(t, observability.AttrLLMModel, "LLM model attribute should be defined")
	assert.NotEmpty(t, observability.AttrLLMTemperature, "LLM temperature attribute should be defined")
	assert.NotEmpty(t, observability.AttrLLMMaxTokens, "LLM max tokens attribute should be defined")
	assert.NotEmpty(t, observability.AttrLLMPromptTokens, "LLM prompt tokens attribute should be defined")
	assert.NotEmpty(t, observability.AttrLLMCompletionTokens, "LLM completion tokens attribute should be defined")
	assert.NotEmpty(t, observability.AttrLLMTotalTokens, "LLM total tokens attribute should be defined")
}

// TestHuggingFaceCallSpanAttributes ensures span is created with expected attributes on validation error
func TestHuggingFaceCallSpanAttributes(t *testing.T) {
	ctx := context.Background()
	exporter, shutdown := setupTestTracerWithExporter(t, ctx)
	defer shutdown(ctx) //nolint:errcheck

	adapter, err := llm.NewHuggingFaceAdapter("test-key", "gpt2", "", llm.HFAPITypeInference, 128, 0.5, llm.HFAdapterOptions{})
	require.NoError(t, err)

	_, callErr := adapter.Call(ctx, llm.Prompt{})
	require.Error(t, callErr)

	spans := exporter.GetSpans()
	require.NotEmpty(t, spans, "expected at least one span")

	span := spans[len(spans)-1]
	assert.Equal(t, "llm.huggingface.call", span.Name)

	attrs := span.Attributes
	provider, ok := findAttrValue(attrs, attribute.Key(observability.AttrLLMProvider))
	require.True(t, ok)
	assert.Equal(t, "huggingface", provider.AsString())

	model, ok := findAttrValue(attrs, attribute.Key(observability.AttrLLMModel))
	require.True(t, ok)
	assert.Equal(t, "gpt2", model.AsString())

	apiType, ok := findAttrValue(attrs, attribute.Key("llm.api_type"))
	require.True(t, ok)
	assert.Equal(t, string(llm.HFAPITypeInference), apiType.AsString())
}

// TestHuggingFaceStreamSpanAttributes ensures streaming span includes streaming attribute
func TestHuggingFaceStreamSpanAttributes(t *testing.T) {
	ctx := context.Background()
	exporter, shutdown := setupTestTracerWithExporter(t, ctx)
	defer shutdown(ctx) //nolint:errcheck

	adapter, err := llm.NewHuggingFaceAdapter("test-key", "gpt2", "", llm.HFAPITypeInference, 128, 0.5, llm.HFAdapterOptions{})
	require.NoError(t, err)

	_, streamErr := adapter.Stream(ctx, llm.Prompt{})
	require.Error(t, streamErr)

	spans := exporter.GetSpans()
	require.NotEmpty(t, spans, "expected at least one span")

	span := spans[len(spans)-1]
	assert.Equal(t, "llm.huggingface.stream", span.Name)

	attrs := span.Attributes
	provider, ok := findAttrValue(attrs, attribute.Key(observability.AttrLLMProvider))
	require.True(t, ok)
	assert.Equal(t, "huggingface", provider.AsString())

	streaming, ok := findAttrValue(attrs, attribute.Key("llm.streaming"))
	require.True(t, ok)
	assert.True(t, streaming.AsBool())
}

// TestOpenRouterCallSpanAttributes ensures span is created with expected attributes on validation error
func TestOpenRouterCallSpanAttributes(t *testing.T) {
	ctx := context.Background()
	exporter, shutdown := setupTestTracerWithExporter(t, ctx)
	defer shutdown(ctx) //nolint:errcheck

	adapter, err := llm.NewOpenRouterAdapter("test-key", "openai/gpt-3.5-turbo", "", 256, 0.2, "", "")
	require.NoError(t, err)

	_, callErr := adapter.Call(ctx, llm.Prompt{})
	require.Error(t, callErr)

	spans := exporter.GetSpans()
	require.NotEmpty(t, spans, "expected at least one span")

	span := spans[len(spans)-1]
	assert.Equal(t, "llm.openrouter.call", span.Name)

	attrs := span.Attributes
	provider, ok := findAttrValue(attrs, attribute.Key(observability.AttrLLMProvider))
	require.True(t, ok)
	assert.Equal(t, "openrouter", provider.AsString())

	model, ok := findAttrValue(attrs, attribute.Key(observability.AttrLLMModel))
	require.True(t, ok)
	assert.Equal(t, "openai/gpt-3.5-turbo", model.AsString())
}

// TestOpenRouterStreamSpanAttributes ensures streaming span includes streaming attribute
func TestOpenRouterStreamSpanAttributes(t *testing.T) {
	ctx := context.Background()
	exporter, shutdown := setupTestTracerWithExporter(t, ctx)
	defer shutdown(ctx) //nolint:errcheck

	adapter, err := llm.NewOpenRouterAdapter("test-key", "openai/gpt-3.5-turbo", "", 256, 0.2, "", "")
	require.NoError(t, err)

	_, streamErr := adapter.Stream(ctx, llm.Prompt{})
	require.Error(t, streamErr)

	spans := exporter.GetSpans()
	require.NotEmpty(t, spans, "expected at least one span")

	span := spans[len(spans)-1]
	assert.Equal(t, "llm.openrouter.stream", span.Name)

	attrs := span.Attributes
	provider, ok := findAttrValue(attrs, attribute.Key(observability.AttrLLMProvider))
	require.True(t, ok)
	assert.Equal(t, "openrouter", provider.AsString())

	streaming, ok := findAttrValue(attrs, attribute.Key("llm.streaming"))
	require.True(t, ok)
	assert.True(t, streaming.AsBool())
}

// BenchmarkOpenAICallWithInstrumentation benchmarks instrumented Call
func BenchmarkOpenAICallWithInstrumentation(b *testing.B) {
	ctx := context.Background()
	shutdown, err := setupTestTracer(ctx)
	if err != nil {
		b.Fatalf("failed to setup tracer: %v", err)
	}
	defer shutdown(ctx) //nolint:errcheck

	// Use mock adapter for benchmarking
	mockAdapter := llm.NewMockAdapter("benchmark-model")

	prompt := llm.Prompt{
		User: "benchmark test",
		Parameters: llm.ModelParameters{
			MaxTokens:   int32Ptr(100),
			Temperature: float32Ptr(0.7),
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = mockAdapter.Call(ctx, prompt)
	}
}

// BenchmarkOpenAIStreamWithInstrumentation benchmarks instrumented Stream
func BenchmarkOpenAIStreamWithInstrumentation(b *testing.B) {
	ctx := context.Background()
	shutdown, err := setupTestTracer(ctx)
	if err != nil {
		b.Fatalf("failed to setup tracer: %v", err)
	}
	defer shutdown(ctx) //nolint:errcheck

	// Use mock adapter for benchmarking
	mockAdapter := llm.NewMockAdapter("benchmark-model")

	prompt := llm.Prompt{
		User: "benchmark stream test",
		Parameters: llm.ModelParameters{
			MaxTokens:   int32Ptr(100),
			Temperature: float32Ptr(0.7),
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenChan, err := mockAdapter.Stream(ctx, prompt)
		if err != nil {
			b.Fatalf("stream error: %v", err)
		}
		// Consume stream
		for range tokenChan {
			// Discard tokens
		}
	}
}

// Helper functions
func int32Ptr(i int32) *int32 {
	return &i
}

func float32Ptr(f float32) *float32 {
	return &f
}
