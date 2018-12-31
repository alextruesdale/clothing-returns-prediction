# Read data
known <- read.csv("BADS_WS1819_known.csv")
unknown <- read.csv("BADS_WS1819_unknown.csv")

# Create model
logit <- glm("return ~ item_price", data = known, family = binomial(link="logit"))
pred_unknown <- predict(logit, newdata = unknown)

prediction <- cbind("order_item_id" = unknown$order_item_id, "return" = pred_unknown)

# Save predictions
write.csv(prediction, file = "example_prediction.csv", row.names = FALSE)
