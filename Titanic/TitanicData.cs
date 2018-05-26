using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Api;

namespace Titanic
{
    public class TitanicData
    {
        //intで定義するとエラーでるのでfloatで定義したほうがよさそう
        [Column(ordinal: "0")]
        public float PassengerId;

        [Column(ordinal: "1", name:"Label")]
        public float Survived;

        [Column(ordinal: "2")]
        public float Pclass;

        [Column(ordinal: "3")]
        public string Name;

        [Column(ordinal: "4")]
        public string Sex;

        [Column(ordinal: "5")]
        public float Age;

        [Column(ordinal: "6")]
        public float SibSp;

        [Column(ordinal: "7")]
        public float Parch;

        [Column(ordinal: "8")]
        public string Ticket;

        [Column(ordinal: "9")]
        public float Fare;

        [Column(ordinal: "10")]
        public string Cabin;

        [Column(ordinal: "11")]
        public string Embarked;
    }

    public class TitanicPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedSurvived;
    }
}
