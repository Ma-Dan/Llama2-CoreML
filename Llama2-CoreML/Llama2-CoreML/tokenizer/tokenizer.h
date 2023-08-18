#ifndef LLAMA2_TOKENIZER_H_
#define LLAMA2_TOKENIZER_H_

#include <stdio.h>
#import <Foundation/Foundation.h>

@interface Tokenizer : NSObject

- (nullable instancetype)initWithTokenizerPath:
(NSString*)tokenizerPath;

- (NSArray<NSNumber *> *)tokenize: (NSString*) prompt;

- (int)sample: (const float*)logits: (float)temperature: (float)topp;

- (NSString*)get_result: (int)token;

@end

#endif
