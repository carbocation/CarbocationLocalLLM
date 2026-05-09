import CarbocationLocalLLM
import CarbocationLocalLLMTools
import Foundation
import XCTest

final class StandardToolsTests: XCTestCase {
    func testCalculateAddsDecimalOperands() throws {
        let output = try LLMCalculateTool.evaluate(arguments: [
            "operation": "add",
            "operands": [0.1, 0.2]
        ])

        XCTAssertEqual(output.value(forKey: "ok"), .bool(true))
        XCTAssertEqual(output.string(forKey: "result"), "0.3")
    }

    func testCalculateReturnsValidationErrorForDivideByZero() async throws {
        let output = try await LLMCalculateTool.tool().call(arguments: [
            "operation": "divide",
            "operands": [10, 0]
        ])

        XCTAssertEqual(output.value(forKey: "ok"), .bool(false))
        XCTAssertEqual(output.value(forKey: "error")?.string(forKey: "code"), "invalid_arguments")
        XCTAssertTrue(output.value(forKey: "error")?.string(forKey: "message")?.contains("Division by zero") == true)
    }

    func testCalculateReturnsValidationErrorForUnsupportedOperation() async throws {
        let output = try await LLMCalculateTool.tool().call(arguments: [
            "operation": "modulo",
            "operands": [10, 3]
        ])

        XCTAssertEqual(output.value(forKey: "ok"), .bool(false))
        XCTAssertTrue(output.value(forKey: "error")?.string(forKey: "message")?.contains("Unsupported operation") == true)
    }

    func testConvertUnitsConvertsMilesToKilometers() throws {
        let output = try LLMConvertUnitsTool.convert(arguments: [
            "category": "length",
            "value": 1,
            "from_unit": "miles",
            "to_unit": "kilometers"
        ])

        XCTAssertEqual(output.value(forKey: "ok"), .bool(true))
        XCTAssertEqual(output.string(forKey: "to_unit"), "kilometers")
        XCTAssertEqual(output.double(forKey: "converted_value") ?? 0, 1.609344, accuracy: 0.000001)
    }

    func testConvertUnitsAcceptsCommonLengthAliases() throws {
        let singularOutput = try LLMConvertUnitsTool.convert(arguments: [
            "category": "length",
            "value": 12,
            "from_unit": "mile",
            "to_unit": "kilometer"
        ])

        XCTAssertEqual(singularOutput.value(forKey: "ok"), .bool(true))
        XCTAssertEqual(singularOutput.string(forKey: "from_unit_canonical"), "miles")
        XCTAssertEqual(singularOutput.string(forKey: "to_unit_canonical"), "kilometers")
        XCTAssertEqual(singularOutput.double(forKey: "converted_value") ?? 0, 19.312128, accuracy: 0.000001)

        let abbreviatedOutput = try LLMConvertUnitsTool.convert(arguments: [
            "category": "length",
            "value": 12,
            "from_unit": "mi",
            "to_unit": "km"
        ])

        XCTAssertEqual(abbreviatedOutput.value(forKey: "ok"), .bool(true))
        XCTAssertEqual(abbreviatedOutput.string(forKey: "from_unit_canonical"), "miles")
        XCTAssertEqual(abbreviatedOutput.string(forKey: "to_unit_canonical"), "kilometers")
        XCTAssertEqual(abbreviatedOutput.double(forKey: "converted_value") ?? 0, 19.312128, accuracy: 0.000001)
    }

    func testConvertUnitsConvertsFahrenheitToCelsius() throws {
        let output = try LLMConvertUnitsTool.convert(arguments: [
            "category": "temperature",
            "value": 32,
            "from_unit": "fahrenheit",
            "to_unit": "celsius"
        ])

        XCTAssertEqual(output.value(forKey: "ok"), .bool(true))
        XCTAssertEqual(output.double(forKey: "converted_value") ?? 1, 0, accuracy: 0.000001)
    }

    func testConvertUnitsRejectsCurrency() async throws {
        let output = try await LLMConvertUnitsTool.tool().call(arguments: [
            "category": "currency",
            "value": 10,
            "from_unit": "usd",
            "to_unit": "eur"
        ])

        XCTAssertEqual(output.value(forKey: "ok"), .bool(false))
        XCTAssertTrue(output.value(forKey: "error")?.string(forKey: "message")?.contains("Unsupported conversion category") == true)
    }

    func testConvertUnitsRejectsInvalidUnit() async throws {
        let output = try await LLMConvertUnitsTool.tool().call(arguments: [
            "category": "length",
            "value": 10,
            "from_unit": "meters",
            "to_unit": "pounds"
        ])

        XCTAssertEqual(output.value(forKey: "ok"), .bool(false))
        XCTAssertTrue(output.value(forKey: "error")?.string(forKey: "message")?.contains("Unsupported unit identifier") == true)
    }

    func testLoadWebpageExtractsTitleAndBodyText() async throws {
        let html = """
        <html><head><title>Example Page</title><script>ignore()</script></head>
        <body><h1>Hello</h1><p>Readable text.</p></body></html>
        """
        let output = try await LLMLoadWebpageTool.load(
            arguments: ["url": "https://example.com"],
            fetcher: StubWebpageFetcher(html: html)
        )

        XCTAssertEqual(output.value(forKey: "ok"), .bool(true))
        XCTAssertEqual(output.string(forKey: "title"), "Example Page")
        XCTAssertEqual(output.string(forKey: "text"), "Hello Readable text.")
        XCTAssertEqual(output.value(forKey: "truncated"), .bool(false))
    }

    func testLoadWebpageAppliesOutputCharacterCap() async throws {
        let longText = String(repeating: "word ", count: 400)
        let output = try await LLMLoadWebpageTool.load(
            arguments: ["url": "https://example.com/long"],
            configuration: LLMLoadWebpageTool.Configuration(maximumOutputCharacters: 1_000),
            fetcher: StubWebpageFetcher(html: "<html><body>\(longText)</body></html>")
        )

        XCTAssertEqual(output.value(forKey: "ok"), .bool(true))
        XCTAssertEqual(output.value(forKey: "truncated"), .bool(true))
        XCTAssertEqual(output.string(forKey: "text")?.count, 1_000)
    }

    func testLoadWebpageNormalizesLiteralSlashEscapesWithoutDecodingPercentEscapes() async throws {
        let expectedURL = "https://www.bing.com/search?q=S%26P+500+closing+price+May+8+2026"
        let fetcher = RecordingWebpageFetcher(
            html: "<html><body>Result</body></html>",
            finalURL: URL(string: expectedURL)!
        )

        let output = try await LLMLoadWebpageTool.load(
            arguments: [
                "url": #"https:\/\/www.bing.com\/search?q=S%26P+500+closing+price+May+8+2026"#
            ],
            fetcher: fetcher
        )

        let requestedURLs = await fetcher.requestedURLs()
        XCTAssertEqual(requestedURLs.map(\.absoluteString), [expectedURL])
        XCTAssertEqual(output.string(forKey: "url"), expectedURL)
        XCTAssertEqual(output.string(forKey: "url")?.contains("%26"), true)
        XCTAssertEqual(output.string(forKey: "url")?.contains("%2526"), false)
    }

    func testLoadWebpageRejectsNonHTTPURL() async throws {
        let output = try await LLMLoadWebpageTool.tool(fetcher: StubWebpageFetcher(html: "")).call(arguments: [
            "url": "file:///etc/passwd"
        ])

        XCTAssertEqual(output.value(forKey: "ok"), .bool(false))
        XCTAssertEqual(output.value(forKey: "error")?.string(forKey: "code"), "unsupported_scheme")
    }

    func testLoadWebpageRejectsHostlessHTTPURLsBeforeFetching() async throws {
        for rawURL in ["https:///path", "https://"] {
            let fetcher = RecordingWebpageFetcher()
            let output = try await LLMLoadWebpageTool.tool(fetcher: fetcher).call(arguments: [
                "url": .string(rawURL)
            ])

            XCTAssertEqual(output.value(forKey: "ok"), .bool(false))
            XCTAssertEqual(output.value(forKey: "error")?.string(forKey: "code"), "invalid_url")
            let requestCount = await fetcher.requestCount()
            XCTAssertEqual(requestCount, 0)
        }
    }

    func testLoadWebpageRejectsInvalidCharactersBeforeURLAutoEncoding() async throws {
        let fetcher = RecordingWebpageFetcher()

        let output = try await LLMLoadWebpageTool.tool(fetcher: fetcher).call(arguments: [
            "url": "https://www.bing.com/search?q=S%26P 500"
        ])

        XCTAssertEqual(output.value(forKey: "ok"), .bool(false))
        XCTAssertEqual(output.value(forKey: "error")?.string(forKey: "code"), "invalid_url")
        XCTAssertEqual(output.value(forKey: "error")?.string(forKey: "message")?.contains("S%26P 500"), true)
        XCTAssertEqual(output.value(forKey: "error")?.string(forKey: "message")?.contains("%2526"), false)
        let requestCount = await fetcher.requestCount()
        XCTAssertEqual(requestCount, 0)
    }

    func testLoadWebpageRejectsHostlessHTTPRedirect() async throws {
        let fetcher = RecordingWebpageFetcher(finalURL: URL(string: "https:///redirect")!)

        let output = try await LLMLoadWebpageTool.tool(fetcher: fetcher).call(arguments: [
            "url": "https://example.com"
        ])

        XCTAssertEqual(output.value(forKey: "ok"), .bool(false))
        XCTAssertEqual(output.value(forKey: "error")?.string(forKey: "code"), "unsupported_redirect")
        let requestCount = await fetcher.requestCount()
        XCTAssertEqual(requestCount, 1)
    }

    func testLoadWebpageReturnsErrorForTimedOutFetcher() async throws {
        let output = try await LLMLoadWebpageTool.tool(fetcher: StubWebpageFetcher(error: URLError(.timedOut))).call(arguments: [
            "url": "https://example.com"
        ])

        XCTAssertEqual(output.value(forKey: "ok"), .bool(false))
        XCTAssertEqual(output.value(forKey: "error")?.string(forKey: "code"), "webpage_load_failed")
    }
}

private struct StubWebpageFetcher: LLMWebpageFetching {
    var html: String
    var errorCode: URLError.Code?
    var finalURL: URL
    var statusCode: Int?

    init(
        html: String = "",
        error: URLError? = nil,
        finalURL: URL = URL(string: "https://example.com")!,
        statusCode: Int? = 200
    ) {
        self.html = html
        self.errorCode = error?.code
        self.finalURL = finalURL
        self.statusCode = statusCode
    }

    func fetch(_ request: URLRequest) async throws -> LLMWebpageFetchResponse {
        if let errorCode {
            throw URLError(errorCode)
        }
        return LLMWebpageFetchResponse(
            data: Data(html.utf8),
            finalURL: finalURL,
            statusCode: statusCode,
            mimeType: "text/html"
        )
    }
}

private actor RecordingWebpageFetcher: LLMWebpageFetching {
    private var requests: [URLRequest] = []
    private let html: String
    private let finalURL: URL
    private let statusCode: Int?

    init(
        html: String = "",
        finalURL: URL = URL(string: "https://example.com")!,
        statusCode: Int? = 200
    ) {
        self.html = html
        self.finalURL = finalURL
        self.statusCode = statusCode
    }

    func fetch(_ request: URLRequest) async throws -> LLMWebpageFetchResponse {
        requests.append(request)
        return LLMWebpageFetchResponse(
            data: Data(html.utf8),
            finalURL: finalURL,
            statusCode: statusCode,
            mimeType: "text/html"
        )
    }

    func requestedURLs() -> [URL] {
        requests.compactMap(\.url)
    }

    func requestCount() -> Int {
        requests.count
    }
}
