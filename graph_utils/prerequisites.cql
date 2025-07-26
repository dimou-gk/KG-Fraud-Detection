//Set Weight (newbalance_x - oldbalance_x)
MATCH (sender)-[r1:MADE_A]->(txn)-[r2:WITH_]->(receiver),
      (txn)-[r3:AT]->(time:Time)
WHERE r1.oldbalanceOrg IS NOT NULL AND r1.newbalanceOrg IS NOT NULL AND
      r2.oldbalanceDest IS NOT NULL AND r2.newbalanceDest IS NOT NULL

WITH r1, r2, r3,
     r1.oldbalanceOrg AS oldOrg, r1.newbalanceOrg AS newOrg,
     r2.oldbalanceDest AS oldDest, r2.newbalanceDest AS newDest

WITH r1, r2, r3,
     (newOrg - oldOrg) AS senderWeight,
     (newDest - oldDest) AS receiverWeight

SET r1.weight = senderWeight,
    r2.weight = receiverWeight,
    r3.weight = receiverWeight - senderWeight


//Normalize Weights
MATCH ()-[r]->()
WHERE r.weight is not null
WITH min(r.weight) AS minWeight

// Step 2: Shift all weights by |minWeight| + epsilon
WITH abs(minWeight) + 0.001 AS shiftAmount

// Step 3: Apply to all relationships
MATCH ()-[r]->()
WHERE r.weight is not null
SET r.normalizedWeight = r.weight + shiftAmount


//Create Projection (For Centralities)
CALL gds.graph.project( 
  'myGraphforCentralitesCalc', 
  {
      Time: {properties: ['hour']},
      User: {},
      Transaction: {properties: ['amount', 'isFraud']}
  },
  {
      MADE_A: {properties: ['newbalanceOrg', 'oldbalanceOrg', 'normalizedWeight']},
      WITH_: {properties: ['newbalanceDest', 'oldbalanceDest', 'normalizedWeight']},
      AT: {properties: 'normalizedWeight'}
  }
)