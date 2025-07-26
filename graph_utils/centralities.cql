//Degree (Unweighted) Centrality
call gds.degree.write(
    'myGraphforCentralitesCalc', 
    {orientation: "REVERSE",
    writeProperty: 'degree_unweighted'}
) 
YIELD centralityDistribution, nodePropertiesWritten
RETURN centralityDistribution.min AS minimumScore, centralityDistribution.mean AS meanScore, nodePropertiesWritten


//Eigenvector (Unweighted) Centrality
call gds.eigenvector.write(
    'myGraphforCentralitesCalc', 
    {maxIterations: 20,
    writeProperty: 'eigenvector_unweighted'}
) 
YIELD centralityDistribution, nodePropertiesWritten
RETURN centralityDistribution.min AS minimumScore, centralityDistribution.mean AS meanScore, nodePropertiesWritten


//Betweenness Centrality
CALL gds.betweenness.write(
    'myGraphforCentralitesCalc',
    {writeProperty: 'betweenness'})
YIELD centralityDistribution, nodePropertiesWritten
RETURN centralityDistribution.min AS minimumScore, centralityDistribution.mean AS meanScore, nodePropertiesWritten


//HITS Centrality
CALL gds.alpha.hits.write(
    'myGraphforCentralitesCalc', {
     hitsIterations: 20,
     authProperty: 'hitsAuth',
     hubProperty: 'hitsHub'})
YIELD nodePropertiesWritten, ranIterations


//Closeness Centrality
call apoc.periodic.iterate(
    "CALL gds.beta.closeness.write('myGraphforCentralitesCalc', {writeProperty: 'closeness'}) 
    YIELD centralityDistribution, nodePropertiesWritten",
    "RETURN centralityDistribution.min AS minimumScore, centralityDistribution.mean AS meanScore, nodePropertiesWritten",
{batchSize:1000, parallel:true}
)


//Harmonic Centrality
CALL gds.closeness.harmonic.write(
    'myGraphforCentralitesCalc',
    {writeProperty: 'score'})
YIELD nodePropertiesWritten


//PageRank Centrality
CALL gds.pageRank.write(
    'myGraphforCentralitesCalc',
    {maxIterations: 20,
     dampingFactor: 0.85,
     writeProperty: 'pagerank'
})
YIELD nodePropertiesWritten, ranIterations


//ArticleRank Centrality
CALL gds.articleRank.write(
    'myGraphforCentralitesCalc', 
    {writeProperty: 'centrality'
})
YIELD nodePropertiesWritten, ranIterations


//CELF Centrality
CALL gds.influenceMaximization.celf.write(
    'myGraphforCentralitesCalc', 
    {writeProperty: 'celfSpread',
     seedSetSize: 3
})
YIELD nodePropertiesWritten


//--------------------------------------------//
//Weighted Centralities Calculation
//Degree (Weighted) Centrality
call gds.degree.write(
    'myGraphforCentralitesCalc', 
    {relationshipWeightProperty: 'normalizedWeight',
    orientation: "REVERSE",
    writeProperty: 'degree_weighted'}
) 
YIELD centralityDistribution, nodePropertiesWritten
RETURN centralityDistribution.min AS minimumScore, centralityDistribution.mean AS meanScore, nodePropertiesWritten


//Eigenvector (Weighted) Centrality
call gds.eigenvector.write(
    'myGraphforCentralitesCalc', 
    {relationshipWeightProperty: 'normalizedWeight',
    maxIterations: 20,
    writeProperty: 'eigenvector_weighted'}
) 
YIELD centralityDistribution, nodePropertiesWritten
RETURN centralityDistribution.min AS minimumScore, centralityDistribution.mean AS meanScore, nodePropertiesWritten