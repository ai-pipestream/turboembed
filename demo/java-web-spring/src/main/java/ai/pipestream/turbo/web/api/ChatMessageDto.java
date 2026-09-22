// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import ai.pipestream.turbo.Message;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;

/** One chat message. The bundle's own chat template turns the list into a prompt. */
@Schema(name = "ChatMessage", description = "One chat turn, rendered through the bundle's chat template")
public record ChatMessageDto(
        @Schema(description = "Role, as the bundle's chat template names it", example = "user")
        @NotBlank(message = "messages[].role must not be blank") String role,
        @Schema(description = "Message text", example = "Summarize the paragraph in two sentences.")
        @NotBlank(message = "messages[].content must not be blank") String content) {

    /** The binding's message record. */
    public Message toTurbo() {
        return new Message(role, content);
    }
}
