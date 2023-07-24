const { Configuration, OpenAIApi } = require("openai");
exports.handler = async function(context, event, callback) {
  const twiml = new Twilio.twiml.MessagingResponse();
  const inbMsg = event.Body.toLowerCase().trim();
  const configuration = new Configuration({
    apiKey: process.env.OPENAI_API_KEY
  });
  const openai = new OpenAIApi(configuration);
  const response = await openai.createCompletion({
      model: "text-davinci-003",
      prompt: inbMsg,
      temperature: 0.7, //A number between 0 and 1 that determines how many creative risks the engine takes when generating text.
      max_tokens: 3000, // Maximum completion length. max: 4000-prompt
      frequency_penalty: 0.7 // # between 0 and 1. higher -> bigger the effort the model will make in not repeating itself.
    });
  twiml.message(response.data.choices[0].text);
  callback(null, twiml);
};